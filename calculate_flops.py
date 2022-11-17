"""
Reference: 
https://aistudio.baidu.com/paddle/forum/topic/show/989606?fbclid=IwAR2R5FPf_iYIGKTOZJOmZzUEoi_-OhNJ63k-zKqEOnqatYXoDvVY0vgUdZI
https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/static/load_inference_model_cn.html
"""

import argparse
import paddleslim as slim
import paddle


def calculateInferenceModelFlops(model_prefix: str):

    paddle.enable_static()
    startup_program = paddle.static.default_startup_program()
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(startup_program)

    [inference_program, feed_target_names,
     fetch_targets] = (paddle.static.load_inference_model(model_prefix, exe))

    flops = slim.analysis.flops(inference_program)

    print(f"Flops: {flops}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate inference model flops.")

    parser.add_argument("--model_prefix",
                        dest="model_prefix",
                        type=str,
                        help="The model prefix. E.g. ./det_db_v2.0/inference",
                        required=True)

    args = parser.parse_args()

    calculateInferenceModelFlops(args.model_prefix)
