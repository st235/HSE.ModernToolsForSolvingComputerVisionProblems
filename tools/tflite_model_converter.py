import argparse
import multiprocessing
import os

import tflite
import tvm
import tvm.testing
from tvm import autotvm
from tvm import relay
from tvm import transform
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner


def __extract_tasks_autotvm(mod, target, params):
    print("Extract tasks...")
    print(mod)
    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params
    )
    assert (len(tasks) > 0)
    return tasks


def __run_tuning_autotvm(
        tasks, measure_option, n_trial=0, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if n_trial == 0:
            n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


def __convert(tfmodel_path: str,
              input_shape: list[int],
              input_tensor: str,
              input_dtype: int,
              target: str,
              run_tuning: bool):
    tflite_model_buf = open(tfmodel_path, "rb").read()
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
    )

    with tvm.transform.PassContext(opt_level=3):
        mod_nchw = tvm.relay.transform.InferType()(mod)
        mod_nchw = tvm.relay.transform.ConvertLayout({"nn.conv2d": ["NCHW", "OIHW"],
                                                      "image.resize2d": ["NCHW"],
                                                      "nn.conv2d_transpose": ["NCHW", "OIHW"]})(mod_nchw)

    if run_tuning:
        log_file = 'tuning_logging.log'
        num_threads = multiprocessing.cpu_count()
        print("Num threads: ", num_threads)
        os.environ["TVM_NUM_THREADS"] = str(num_threads)

        tuning_option = {
            "log_filename": log_file,
            "tuner": "xgb",
            "early_stopping": False,
            "n_trial": 1,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(),
                runner=autotvm.LocalRunner(
                    number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
                ),
            ),
        }

        tasks = __extract_tasks_autotvm(mod_nchw, target, params)
        __run_tuning_autotvm(tasks, **tuning_option)

        with autotvm.apply_history_best(log_file):
            with transform.PassContext(opt_level=3):
                lib = relay.build(mod_nchw, target=target, params=params)
                lib.export_library('./model.so')
    else:
        with transform.PassContext(opt_level=3):
            lib = relay.build(mod_nchw, target, params=params)
            lib.export_library('./model.so')


def parse_arguments():
    parser = argparse.ArgumentParser(description='TFLite model converter script')
    parser.add_argument('file', type=str, help='TFLite model file.')
    parser.add_argument('--input_shape', '-is', nargs='+', type=int, help='Input shape as array.',
                        required=True)
    parser.add_argument('--input_tensor', type=str, default='input', help='Input tensor name.')
    parser.add_argument('--input_dtype', type=str, default='float32', help='Input data type.')
    parser.add_argument('--target', type=str, help='Optimisation target.',
                        required=True)
    parser.add_argument('--run_tuning', action='store_true', help='Indicates whether the script should run tuning.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    __convert(tfmodel_path=args.file,
              input_shape=args.input_shape,
              input_tensor=args.input_tensor,
              input_dtype=args.input_dtype,
              target=args.target,
              run_tuning=args.run_tuning)


if __name__ == '__main__':
    main()
