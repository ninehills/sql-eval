from warnings import warn


try:
    import qianfan
    import qianfan.errors
except ImportError as e:
    warn("Please install qianfan from PyPI: pip install qianfan")
    raise e

from concurrent.futures import ThreadPoolExecutor, as_completed
from eval.eval import compare_query_results
import pandas as pd
from utils.pruning import generate_prompt
from utils.questions import prepare_questions_df
from tqdm import tqdm
from time import perf_counter as pc


def process_row(row, completion: qianfan.Completion, timeout_gen: int, timeout_exec: int):
    start_time = pc()
    try:
        resp = completion.do(
            prompt=row["prompt"],
            retry_count=2,
            request_timeout=timeout_gen
        )
    except qianfan.errors.RequestTimeoutError as e:
        row["error_msg"] = f"QUERY GENERATION TIMEOUT: {e.__class__.__name__}: {e}"
    except qianfan.errors.QianfanError as e:
        row["error_msg"] = f"QUERY GENERATION ERROR: {e.__class__.__name__}: {e}"
    end_time = pc()

    text = resp.get("body", {}).get("result", "")
    if not text:
        row["error_msg"] = "QUERY GENERATION ERROR: No result"
    sql_split =  text.split('```sql')
    if len(sql_split) > 1:
        text = sql_split[1]
    generated_query = text.split('```')[0].split(";")[0].strip() + ";"
    if generated_query == ";":
        row["error_msg"] = "QUERY GENERATION ERROR: Empty query"

    end_time = pc()
    row["generated_query"] = generated_query
    row["latency_seconds"] = end_time - start_time
    golden_query = row["query"]
    db_name = row["db_name"]
    question = row["question"]
    query_category = row["query_category"]
    exact_match = correct = 0

    db_creds = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "database": db_name,
    }

    try:
        exact_match, correct = compare_query_results(
            query_gold=golden_query,
            query_gen=generated_query,
            db_name=db_name,
            db_creds=db_creds,
            question=question,
            query_category=query_category,
            timeout=timeout_exec,
        )
        row["exact_match"] = int(exact_match)
        row["correct"] = int(correct)
        row["error_msg"] = ""
    except Exception as e:
        row["error_db_exec"] = 1
        row["error_msg"] = f"QUERY EXECUTION ERROR: {e}"

    return row

def run_qianfan_eval(args):
    # get params from args
    questions_file = args.questions_file
    prompt_file_list = args.prompt_file
    num_questions = args.num_questions
    output_file_list = args.output_file
    max_workers = args.parallel_threads
    timeout_gen = args.timeout_gen
    timeout_exec = args.timeout_exec
    completion = qianfan.Completion(
        model = args.model,
        query_per_second=1)

    print("preparing questions...")
    # get questions
    print(f"Using {num_questions} questions from {questions_file}")
    df = prepare_questions_df(questions_file, num_questions)

    for prompt_file, output_file in zip(prompt_file_list, output_file_list):
        # create a prompt for each question
        df["prompt"] = df[["question", "db_name", "instructions"]].apply(
            lambda row: generate_prompt(
                prompt_file, row["question"], row["db_name"], row["instructions"]
            ),
            axis=1,
        )

        print("questions prepared\nnow loading chain...")
        # initialize tokenizer and model
        total_tried = 0
        total_correct = 0
        output_rows = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in df.to_dict("records"):
                futures.append(executor.submit(process_row, row, completion, timeout_gen, timeout_exec))

            with tqdm(as_completed(futures), total=len(futures)) as pbar:
                for f in pbar:
                    row = f.result()
                    output_rows.append(row)
                    if row["correct"]:
                        total_correct += 1
                    total_tried += 1
                    pbar.update(1)
                    pbar.set_description(
                        f"Correct so far: {total_correct}/{total_tried} ({100*total_correct/total_tried:.2f}%)"
                    )

        output_df = pd.DataFrame(output_rows)
        del output_df["prompt"]
        print(output_df.groupby("query_category")[["exact_match", "correct"]].mean())
        output_df = output_df.sort_values(by=["db_name", "query_category", "question"])
        try:
            output_df.to_csv(output_file, index=False, float_format="%.2f")
        except:
            output_df.to_pickle(output_file)

        # get average rate of correct results
        avg_subset = output_df["correct"].sum() / len(output_df)
        print(f"Average correct rate: {avg_subset:.2f}")