import subprocess

model_parameters = ['GPT4o']

# Dataset and parameters setting
tasks = [
    {
        'data_task': 'Emotion',
        'problem_task': 'Classification',
        'data': 'emobench',
        'TQ': ['emobench-Clear', 'emobench-EmDe', 'emobench-Ana'],
        'CT': 'emobench',
        'OI': 'emobench',
        'SI': 'persona-none',
        'PS_base': 'emobench'
    },
    {
        'data_task': 'Emotion',
        'problem_task': 'Classification',
        'data': 'goemotion',
        'TQ': ['goemotion-Clear', 'goemotion-EmDe', 'goemotion-Ana'],
        'CT': 'goemotion',
        'OI': 'goemotion',
        'SI': 'persona-none',
        'PS_base': 'goemotion'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'dreaddit',
        'TQ': ['dreaddit-Clear', 'dreaddit-EmDe', 'dreaddit-Ana'],
        'CT': 'dreaddit',
        'OI': 'dreaddit',
        'SI': 'persona-none',
        'PS_base': 'dreaddit'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'cssrs',
        'TQ': ['cssrs-Clear', 'cssrs-EmDe', 'cssrs-Ana'],
        'CT': 'cssrs',
        'OI': 'cssrs',
        'SI': 'persona-none',
        'PS_base': 'cssrs'
    },
    {
        'data_task': 'Mental-Health',
        'problem_task': 'Classification',
        'data': 'sdcnl',
        'TQ': ['sdcnl-Clear', 'sdcnl-EmDe', 'sdcnl-Ana'],
        'CT': 'sdcnl',
        'OI': 'sdcnl',
        'SI': 'persona-none',
        'PS_base': 'sdcnl'
    }
]


def generate_commands(max_rows=200):
    commands = []
    for task in tasks:
        for TQ in task['TQ']:
            for shot in range(0,1):  # shot 0~3
                # When the shot is 0, set PS to '{dataset}-none'; when the shot is 1 or more, set it to '{dataset}-fewshot_icl'.
                PS = f"{task['PS_base']}-none" if shot == 0 else f"{task['PS_base']}-fewshot_icl"

                commands.append([
                    '--data_task', task['data_task'],
                    '--problem_task', task['problem_task'],
                    '--data', task['data'],
                    '--SI', task['SI'],
                    '--TQ', TQ,
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
