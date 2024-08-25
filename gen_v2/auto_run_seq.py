import subprocess

# 모델 파라미터 설정 (여러 모델을 지정 가능)
model_parameters = ['Ollama_Llama']

# 명령어와 인수 정의 (모델을 제외한 다른 모든 파라미터)
commands = [
[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-none',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-none',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-none',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-fewshot_icl',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-none',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-fewshot_icl',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-none',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-fewshot_icl',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### expert
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-expert',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-none',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-expert',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-fewshot_icl',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-expert',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-fewshot_icl',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'emobench',
        '--SI', 'persona-expert',
        '--TQ', 'emobench_stmliX',
        '--PS', 'emobench-fewshot_icl',
        '--CT', 'emobench',
        '--LD', 'none',
        '--OI', 'emobench',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### goemotion
    ### none
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-none',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-none',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-none',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-none',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-none',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### expert
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-expert',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-none',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-expert',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-expert',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Emotion',
        '--problem_task', 'Classification',
        '--data', 'goemotion',
        '--SI', 'persona-expert',
        '--TQ', 'goemotion_stmliX',
        '--PS', 'goemotion-fewshot_icl',
        '--CT', 'goemotion',
        '--LD', 'none',
        '--OI', 'goemotion',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],

    ### dreaddit
    ### none
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-none',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-none',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-none',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-fewshot_icl',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-none',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-fewshot_icl',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-none',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-fewshot_icl',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### expert
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-expert',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-none',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-expert',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-fewshot_icl',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-expert',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-fewshot_icl',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'dreaddit',
        '--SI', 'persona-expert',
        '--TQ', 'dreaddit_stmliX',
        '--PS', 'dreaddit-fewshot_icl',
        '--CT', 'dreaddit',
        '--LD', 'none',
        '--OI', 'dreaddit',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### cssrs
    ### none
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-none',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-none',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-none',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-fewshot_icl',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-none',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-fewshot_icl',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-none',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-fewshot_icl',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### expert
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-expert',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-none',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-expert',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-fewshot_icl',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-expert',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-fewshot_icl',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'cssrs',
        '--SI', 'persona-expert',
        '--TQ', 'cssrs_stmliX',
        '--PS', 'cssrs-fewshot_icl',
        '--CT', 'cssrs',
        '--LD', 'none',
        '--OI', 'cssrs',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### sdcnl
    ### none
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-none',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-none',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-none',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-fewshot_icl',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-none',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-fewshot_icl',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-none',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-fewshot_icl',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    ### expert
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-expert',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-none',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '0',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],
    [
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-expert',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-fewshot_icl',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '1',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-expert',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-fewshot_icl',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '2',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],[
        '--data_task', 'Mental-Health',
        '--problem_task', 'Classification',
        '--data', 'sdcnl',
        '--SI', 'persona-expert',
        '--TQ', 'sdcnl_stmliX',
        '--PS', 'sdcnl-fewshot_icl',
        '--CT', 'sdcnl',
        '--LD', 'none',
        '--OI', 'sdcnl',
        '--shot', '3',
        '--max_rows', '5',
        '--output_structure', 'index'
    ],

]

def run_command_for_model(model):
    for command in commands:
        full_command = ['python', 'systematic_evaluation.py', '--models', model] + command
        print(f"Running command for model {model}: {' '.join(full_command)}")

        try:
            # subprocess.run을 사용하여 명령을 실행하고, 명령이 완료될 때까지 대기합니다.
            result = subprocess.run(full_command, check=True, text=True, capture_output=True)

            # 출력 결과를 표시합니다.
            print(f"Standard Output for {model}:\n{result.stdout}")
            print(f"Standard Error for {model}:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            # 명령이 실패한 경우 에러 메시지를 표시하고 종료합니다.
            print(f"Error occurred while running model {model}.\nError Message: {e.stderr}")
            return e.returncode

    return 0

# 모델을 순차적으로 실행합니다.
def run_all_models():
    for model in model_parameters:
        result = run_command_for_model(model)
        if result != 0:
            break  # 오류가 발생한 경우 이후 모델 실행을 중단

if __name__ == "__main__":
    run_all_models()
