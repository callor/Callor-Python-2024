import json
import os
import os.path as osp
import time
from typing import List, Dict, Union
from module.llm import get_response_from_llm, extract_json_between_markers

import requests

# 이 모듈은 함수를 래핑하여 어떤 조건이 충족될 때까지 재시도되도록 하는 데 사용할 수 있는 함수 데코레이터를 제공합니다
import backoff

S2_API_KEY = os.getenv("S2_API_KEY")

idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>


당신이 이미 생각해 낸 아이디어는 다음과 같습니다.:

'''
{prev_ideas_string}
'''

제공된 코드로 실현 가능하게 조사할 수 있는 연구 실험 및 방향에 대한 다음의 영향력 있고 창의적인 아이디어를 생각해 보세요.
추가 리소스나 데이터 세트에 액세스할 수 없다는 점에 유의하세요.
모든 아이디어가 특정 훈련 데이터 세트나 모델에 과적합되지 않고 더 광범위한 의미를 갖도록 하세요.

다음 형식으로 응답하세요:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

<THOUGHT>에서 먼저 아이디어에 대한 직관과 동기를 간략하게 논의합니다. 
상위 계획, 필요한 디자인 선택 및 실험의 이상적인 결과를 자세히 설명합니다. 
아이디어가 기존 아이디어와 어떻게 다른지 정당화합니다.

다음 필드를 사용하여 JSON 형식으로 새로운 아이디어를 제공하세요.:
- "Name": 아이디어의 축약된 설명자. 소문자, 공백 없음, 밑줄 허용.
- "Title": 아이디어의 제목은 보고서 작성에 사용됩니다.
- "Experiment": 구현 개요. 예를 들어 어떤 기능을 추가하거나 수정해야 하는지, 어떻게 결과를 얻을 것인지 등
- "Interestingness": 1~10점(가장 낮은 등급부터 가장 높은 등급까지)
- "Feasibility": 1~10점(가장 낮은 등급부터 가장 높은 등급까지)
- "Novelty": 1~10점(가장 낮은 등급부터 가장 높은 등급까지)

평가할때는 신중하고 현실적으로 해야 하며, 이 JSON 은 구문형식이 정확해야 함아이디어를 반복할 수 있으나, 모두 사용할 필요는 없음"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
생각 속에서, 먼저 당신이 방금 창조한 아이디어의 질, 참신성, 실현 가능성을 신중하게 고려해 보세요.

아이디어를 평가하는 데 중요하다고 생각되는 다른 요소가 있다면 포함하세요.

아이디어가 명확하고 간결하며 JSON이 올바른 형식인지 확인하세요.

일을 너무 복잡하게 만들지 마세요.

다음 시도에서는 아이디어를 더욱 다듬고 개선해 보세요.

눈에 띄는 문제가 없는 한 원래 아이디어의 정신을 고수하세요.

이전과 같은 형식으로 응답하세요:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

개선할 것이 없다면, 단순히 이전 JSON을 생각 바로 뒤에 반복하고, 생각의 끝에 JSON 앞에 "I am done"을 포함하세요.
더 이상 변경하지 않을 경우에만 "I am done"을 포함하세요."""

# GENERATE IDEAS
def generate_ideas(
    base_dir,
    client,
    model,
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "rt",encoding="UTF-8") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    # with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
    with open(osp.join(base_dir, "seed_ideas.json"), 'rt', encoding='UTF8') as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea,ensure_ascii=False))

    with open(osp.join(base_dir, "experiment.py"),  'rt', encoding='UTF8') as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"),  'rt', encoding='UTF8') as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print(json_output)

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                        json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print(json_output)

                    if "I am done" in text:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output,ensure_ascii=False))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w",encoding="UTF-8-sig") as f:
        json.dump(ideas, f, indent=4,ensure_ascii=False)

    return ideas


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
    base_dir,
    client,
    model,
    prev_idea_archive=[],
    num_reflections=5,
    max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea,ensure_ascii=False))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.ite
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                            json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w",encoding='UTF-8') as f:
        json.dump(idea_archive, f, indent=4,ensure_ascii=False)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10) -> Union[None, List[Dict]]:
    if not query:
        return None
    rsp = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        headers={"X-API-KEY": S2_API_KEY},
        params={
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
        },
    )
    print(f"Response Status Code: {rsp.status_code}")
    print(
        f"Response Content: {rsp.text[:500]}"
    )  # Print the first 500 characters of the response content
    rsp.raise_for_status()
    results = rsp.json()
    total = results["total"]
    time.sleep(1.0)
    if not total:
        return None

    papers = results["data"]
    return papers


novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


def check_idea_novelty(
    ideas,
    base_dir,
    client,
    model,
    max_num_iterations=10,
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "rt",encoding="UTF-8") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w",encoding="UTF-8") as f:
        json.dump(ideas, f, indent=4,ensure_ascii=False)

    return ideas


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 3
    NUM_REFLECTIONS = 5
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        # default="gpt-4o-2024-08-06",
        default="gpt-4o-2024-05-13",
        choices=[
            "gpt-4o-2024-08-06",
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
        ],
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    args = parser.parse_args()

    # Create client
    if args.model == "gpt-4o-2024-05-13":
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()


    elif args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {args.model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock()
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Vertex AI with model {client_model}.")
        client = anthropic.AnthropicVertex()
    elif args.model == "gpt-4o-2024-08-06" :
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        client_model = "gpt-4o-2024-08-06"
        client = openai.OpenAI()
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com"
        )
    elif args.model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    # base_dir = osp.join("templates", args.experiment)
    base_dir = osp.join(".")
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if args.check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )