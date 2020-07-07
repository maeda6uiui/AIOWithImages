import subprocess

class TrainAndTestDriver(object):
    def __init__(
        self,system_name="jaqket_baseline",
        data_dir="../Data/",
        model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking",
        task_name="jaqket",
        output_dir="./working_dir/output_dir/"):
        self.command=(
            "python3 ../system/{system_name}.py\n"
            "--data_dir {data_dir}\n"
            "--model_name_or_path {model_name_or_path}\n"
            "--task_name {task_name}\n"
            "--entities_fname candidate_entities.json.gz\n"
            "--train_fname train_questions.json\n"
            "--dev_fname dev1_questions.json\n"
            "--test_fname dev2_questions.json\n"
            "--output_dir {output_dir}\n"
            "--train_num_options 4\n"
            "--do_train\n"
            "--do_eval\n"
            "--do_test\n"
            "--per_gpu_train_batch_size 1\n"
            "--gradient_accumulation_steps 8\n"
            "--num_train_epochs 5\n"
            "--logging_steps 10\n"
            "--save_steps 1000"
        ).format(
            system_name=system_name,
            data_dir=data_dir,
            model_name_or_path=model_name_or_path,
            task_name=task_name,
            output_dir=output_dir
        )
        self.command=self.command.split("\n")

        self.command2=(
            "python3 ../system/{system_name}.py\n"
            "--data_dir {data_dir}\n"
            "--dev_fname dev1_questions.json\n"
            "--test_fname dev2_questions.json\n"
            "--task_name {task_name}\n"
            "--entities_fname candidate_entities.json.gz\n"
            "--model_name_or_path {output_dir}\n"
            "--eval_num_options 20\n"
            "--per_gpu_eval_batch_size 4\n"
            "--do_test\n"
            "--do_eval"
        ).format(
            system_name=system_name,
            data_dir=data_dir,
            task_name=task_name,
            output_dir=output_dir
        )
        self.command2=self.command2.split("\n")

    def run(self):
        res=subprocess.check_output(self.command)
        print(res)

        res2=subprocess.check_output(self.command2)
        print(res2)
