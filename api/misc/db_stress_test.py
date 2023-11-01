
import statistics
import time
import random

from pathlib import Path

from api.util.dbtool import DbTool

DBPATH = None


def add_images():
    with DbTool(DBPATH) as db:
        image_list = db.getTrainingImages()
    
    if image_list:
        return

    good = []
    for p in Path("../data/bottle-sealing-surface/good").glob("*.png"):
        good.append(str(p))

    bad = []
    for p in Path("../data/bottle-sealing-surface/bad").glob("*.png"):
        bad.append(str(p))
    print(good, bad)
    
    with DbTool(DBPATH) as db:
        res = db.addTrainingImages(good, 'bottle', 'good')
        print(res)
    with DbTool(DBPATH) as db:
        res = db.addTrainingImages(bad, 'bottle', 'bad')
        print(res)


def train_models(num_models: int):
    with DbTool(DBPATH) as db:
        model_list = db.getModels()
    model_len = len(model_list)

    models_to_train = num_models - model_len
    if models_to_train <= 0:
        print(model_list)
        return
    
    with DbTool(DBPATH) as db:
        _, im_list = db.getDataset('bottle')
    
    for i in range(models_to_train):
        with DbTool(DBPATH) as db:
            # insert model path
            i_id = db.addModel(
                f'fakepath/abc/xyz/bottle_{i+model_len}',
                l_image=im_list,
                model_type='patchcore',  # no check, assume patchcore
                tag=f'bottle_{i+model_len}',
                d_param=None,
                version=1,
            )
            print('model_id', i_id)


def serve_model(model):
    model_id = model['model_id']
    with DbTool(DBPATH) as db:
        db.updateInferencePath(model_id, f'fakepath/servedpath/{model_id}')
        db.addServing(model_id, None)

    return model_id


def run_stress_test(length, record_every=100, wait_time=0.005, print_updates=False):
    with DbTool(DBPATH) as db:
        model_list = db.getModels()

    curr_model = random.choice(model_list)
    curr_model_id = serve_model(curr_model)
    insertion_times = []
    query_times = []
    for i in range(length):
        t = time.perf_counter()
        with DbTool(DBPATH) as db:
            db.addInference(
                curr_model_id,
                f'fakepath/aaaaaaaaaaaaaaaaaa/bbbbbbbbbbbbbbb/ccccccccccccccccccccccc/{i}.png',
                {
                    'pred_score': random.uniform(0, 10),
                    'image_threshold': random.uniform(3, 7),
                    'pred_score_norm': random.random(),
                    'image_threshold_norm': random.random(),
                },
                f'fakepath/aaaaaaaaaaaaaaaaaaaa/bbbbbbbbbbbbb/ccccccccccccccccc/inference_results/images/prediction/{i}.png'
            )
        i_t_f = time.perf_counter() - t
        time.sleep(wait_time)       # mimic waiting period
        if i % (record_every/10) == 0:
            t = time.perf_counter()
            with DbTool(DBPATH) as db:
                res = db.getInference(random.randint(1, i+1))
            g_t_f = time.perf_counter() - t

        if i % record_every == 0:
            if print_updates and i % (record_every*10) == 0:
                print(i, round(i_t_f, 6), round(g_t_f, 6))
            insertion_times.append(i_t_f)
            query_times.append(g_t_f)
            curr_model = random.choice(model_list)
            curr_model_id = serve_model(curr_model)

    return insertion_times, query_times


def print_stats(data, typ: str):
    print('==================================')
    typ = typ.upper()
    print(f'{typ} total data points', len(data))
    print(f'Max time taken for {typ} type {max(data):.6f}')
    print(f'Min time taken for {typ} type {min(data):.6f}')
    print(f'Std. dev. of times for {typ} type {statistics.stdev(data):.6f}')
    print(f'Mean time taken for {typ} type {statistics.mean(data):.6f}')
    print('----------------------------------')
    first_100 = data[:100]
    print(f'Max time taken for first 100 {typ} type {max(first_100):.6f}')
    print(f'Min time taken for first 100 {typ} type {min(first_100):.6f}')
    print(f'Std. dev. of times for first 100 {typ} type {statistics.stdev(first_100):.6f}')
    print(f'Mean time taken for first 100 {typ} type {statistics.mean(first_100):.6f}')
    print('----------------------------------')
    last_100 = data[-100:]
    print(f'Max time taken for last 100 {typ} type {max(last_100):.6f}')
    print(f'Min time taken for last 100 {typ} type {min(last_100):.6f}')
    print(f'Std. dev. of times for last 100 {typ} type {statistics.stdev(last_100):.6f}')
    print(f'Mean time taken for last 100 {typ} type {statistics.mean(last_100):.6f}')
    print('==================================')


if __name__ == '__main__':
    DBPATH = 'misc/stress_test.db'
    ## execute this script from .../hamacho/api folder.
    add_images()
    train_models(7)
    its, qts = run_stress_test(1_800_000, wait_time=0, print_updates=True)
    print_stats(its, 'insertion')
    print_stats(qts, 'query')
