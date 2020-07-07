import os
import subprocess

WORKING_DIR="./working_dir/"

class TrainAndTestScriptGenerator(object):
    def __init__(
        self,system_name="jaqket_baseline",
        data_dir="../Data/",
        model_name_or_path="cl-tohoku/bert-base-japanese-whole-word-masking",
        output_dir=WORKING_DIR+"output_dir/"):
        self.system_name=system_name

        self.command=(
            "python3 ../system/{system_name}.py \\\n"
            "\t--data_dir {data_dir} \\\n"
            "\t--model_name_or_path {model_name_or_path} \\\n"
            "\t--task_name {system_name} \\\n"
            "\t--entities_fname candidate_entities.json.gz \\\n"
            "\t--train_fname train_questions.json \\\n"
            "\t--dev_fname dev1_questions.json \\\n"
            "\t--test_fname dev2_questions.json \\\n"
            "\t--output_dir {output_dir} \\\n"
            "\t--train_num_options 4 \\\n"
            "\t--do_train \\\n"
            "\t--do_eval \\\n"
            "\t--do_test \\\n"
            "\t--per_gpu_train_batch_size 1 \\\n"
            "\t--gradient_accumulation_steps 8 \\\n"
            "\t--num_train_epochs 5 \\\n"
            "\t--logging_steps 10 \\\n"
            "\t--save_steps 1000\n"
        ).format(
            system_name=system_name,
            data_dir=data_dir,
            model_name_or_path=model_name_or_path,
            output_dir=output_dir
        )

        self.command2=(
            "python3 ../system/{system_name}.py \\\n"
            "\t--data_dir {data_dir} \\\n"
            "\t--dev_fname dev1_questions.json \\\n"
            "\t--test_fname dev2_questions.json \\\n"
            "\t--task_name {system_name} \\\n"
            "\t--entities_fname candidate_entities.json.gz \\\n"
            "\t--model_name_or_path {output_dir} \\\n"
            "\t--eval_num_options 20 \\\n"
            "\t--per_gpu_eval_batch_size 4 \\\n"
            "\t--do_test \\\n"
            "\t--do_eval\n"
        ).format(
            system_name=system_name,
            data_dir=data_dir,
            output_dir=output_dir
        )

    def save_script(self):
        script_filename="run_{}_all.sh".format(self.system_name)
        with open(WORKING_DIR+script_filename,"w",encoding="utf-8") as w:
            w.write(self.command)
            w.write("\n")
            w.write(self.command2)

        chmod_command="chmod +x {}".format(WORKING_DIR+script_filename)
        subprocess.run(chmod_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)

    def run(self):
        cd_command="cd ./working_dir/"
        subprocess.run(cd_command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
