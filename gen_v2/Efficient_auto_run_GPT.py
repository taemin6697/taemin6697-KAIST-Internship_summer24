import subprocess

# 모델 파라미터 설정 (여러 모델을 지정 가능)
model_parameters = ['GPT4o']

# 데이터셋과 기타 파라미터 설정
tasks = [
    {
        'data_task': 'Emotion',
        'problem_task': 'Classification',
        'data': 'emobench',
        'TQ': 'emobench_stmliX',
        'CT': 'emobench',
        'OI': 'emobench',
        'SI_options': ['persona-none', 'persona-expert'],
        'PS_base': 'emobench'
    },
    {
        'data_task': 'Emotion',
        'problem_task': 'Classification',
        'data': 'goemotion',
        'TQ': 'goemotion_stmliX',
        'CT': 'goemotion',
        'OI': 'goemotion',
        'SI_options': ['persona-none', 'persona-expert'],
        'PS_base': 'goemotion'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'dreaddit',
        'TQ': 'dreaddit_stmliX',
        'CT': 'dreaddit',
        'OI': 'dreaddit',
        'SI_options': ['persona-none', 'persona-expert'],
        'PS_base': 'dreaddit'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'cssrs',
        'TQ': 'cssrs_stmliX',
        'CT': 'cssrs',
        'OI': 'cssrs',
        'SI_options': ['persona-none', 'persona-expert'],
        'PS_base': 'cssrs'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'sdcnl',
        'TQ': 'sdcnl_stmliX',
        'CT': 'sdcnl',
        'OI': 'sdcnl',
        'SI_options': ['persona-none', 'persona-expert'],
        'PS_base': 'sdcnl'
    }
]


def generate_commands(max_rows=200):
    commands = []
    for task in tasks:
        for SI in task['SI_options']:
            for shot in range(4,6):  # shot 0~3
                # shot이 0일 때는 PS를 '{dataset}-none'으로, shot이 1 이상일 때는 '{dataset}-fewshot_icl'로 설정
                PS = f"{task['PS_base']}-none" if shot == 0 else f"{task['PS_base']}-fewshot_icl"

                commands.append([
                    '--data_task', task['data_task'],
                    '--problem_task', task['problem_task'],
                    '--data', task['data'],
                    '--SI', SI,
                    '--TQ', task['TQ'],
                    '--PS', PS,
                    '--CT', task['CT'],
                    '--LD', 'none',
                    '--OI', task['OI'],
                    '--shot', str(shot),
                    '--max_rows', str(max_rows),
                    '--output_structure', 'index'
                ])
    return commands


def run_command_for_model(model, commands):
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
    commands = generate_commands(max_rows=200)  # 원하는 max_rows 값으로 변경 가능
    for model in model_parameters:
        result = run_command_for_model(model, commands)
        if result != 0:
            break  # 오류가 발생한 경우 이후 모델 실행을 중단


if __name__ == "__main__":
    run_all_models()
