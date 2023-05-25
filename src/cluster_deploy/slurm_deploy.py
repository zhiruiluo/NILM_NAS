import ast
import inspect
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from simple_parsing import Serializable, parse
from pyinstrument import Profiler

template_file = "src/cluster_deploy/sbatch_template.sh"


@dataclass
class SlurmConfig(Serializable):
    exp_name: str = "test_exp"
    job_name: str = field(init=False)
    num_nodes: int = 1
    num_gpus: int = 0
    partition: str = ""
    node_list: str = ""
    load_env: str = ""
    command: str = field(init=False)
    command_suffix: str = ""
    log_dir: str = "logging/"
    exp_dir: str = field(init=False)
    node_info: str = field(init=False)
    slurm_path: str = field(init=False)

    def __post_init__(self):
        if self.node_list is not None and self.node_list != "":
            self.node_info = f"#SBATCH -w {self.node_list}"
        else:
            self.node_info = ""
        self.job_name = "{}_{}".format(self.exp_name, time.strftime("%m%d_%H%M", time.localtime()))
        self.exp_dir = Path(self.log_dir).joinpath(self.job_name)
        # self.command_suffix = self._replace_command(self.command_suffix)
        self.slurm_path = Path(self.exp_dir)
        self.command = f'python -u {self.slurm_path.joinpath("launch.py")}'

    def _replace_command(self, command_str):
        command_str = command_str.replace("{{EXP_NAME}}", self.job_name)
        return command_str
    
    def replace_patterns(self) -> None:
        self.command_suffix = self._replace_command(self.command_suffix)


class SlurmDeploy:
    def __init__(self, func, config) -> None:
        self.args = self._setup_args(config)
        self.func = func
    
    def _setup_args(self, config: SlurmConfig) -> SlurmConfig:
        args: SlurmConfig = parse(SlurmConfig, default=config)
        args.replace_patterns()
        return args
    
    def _extract_func_def(self, source: str):
        node = ast.parse(source=source)
        n1 = node.body[0]
        n1.decorator_list = []
        return ast.unparse(n1)
    
    def _modify_func_decorator(self, module_source: str, func_source: str, modify_func: str):
        node = ast.parse(source=module_source)
        pop_idx = -1
        for i, b in enumerate(node.body):
            if (ast.unparse(b) == ast.unparse(ast.parse(func_source))):            
                pop_idx = i
                break
        node.body[pop_idx] = ast.parse(modify_func)
        return ast.unparse(node)

    # def _modify_func_decorator(self, module_source: str, func_source: str):
    #     module_source = module_source.splitlines(True)
    #     modified_func_source = self._extract_func_def(func_source)
    #     func_source = func_source.splitlines(True)
    #     start = 0
    #     end = 0
    #     for i, l in enumerate(module_source):
    #         if l == func_source[0]:
    #             start = i
    #         if l == func_source[-1]:
    #             end = i
        
    #     modified_source = ''.join(module_source[:start]) + modified_func_source + ''.join(module_source[end+1:])
        
    #     return modified_source

    def _generate_file(self):
        module_source = inspect.getsource(inspect.getmodule(self.func))
        func_source = inspect.getsource(self.func)
        modified_func_source = self._extract_func_def(func_source)
        modifyied_source = self._modify_func_decorator(module_source, func_source, modified_func_source)
        
        # func_source = (
        #     "import sys\nsys.path.append('.')\n" + func_source + "\n\n" + f"{self.func.__name__}()"
        # )
        # modifyied_source = autopep8.fix_code(modifyied_source)
        # fn = args.slurm_path.joinpath(Path(self.func.__code__.co_filename).name)
        
        fn = self.args.slurm_path.joinpath("launch.py")
        with open(fn, "w") as fp:
            fp.write(modifyied_source)

    def _modify_template(self):
        args = self.args
        # ===== Modified the template script =====
        with open(template_file) as f:
            text = f.read()
        text = text.replace("{{JOB_NAME}}", args.job_name)
        text = text.replace("{{JOB_DIR}}", str(args.slurm_path))
        text = text.replace("{{NUM_NODES}}", str(args.num_nodes))
        text = text.replace("{{NUM_GPUS_PER_NODE}}", str(args.num_gpus))
        text = text.replace("{{PARTITION_NAME}}", str(args.partition))
        text = text.replace("{{COMMAND_PLACEHOLDER}}", str(args.command))
        text = text.replace("{{LOAD_ENV}}", str(args.load_env))
        text = text.replace("{{GIVEN_NODE}}", args.node_info)
        text = text.replace("{{COMMAND_SUFFIX}}", args.command_suffix)
        text = text.replace(
            "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO " "PRODUCTION!",
            "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE " "RUNNABLE!",
        )

        # ===== Save the script =====
        script_file = args.slurm_path.joinpath(f"{args.job_name}.sh")
        with open(script_file, "w") as f:
            f.write(text)

        return script_file

    def _prepare_script(self):
        if not self.args.slurm_path.exists():
            self.args.slurm_path.mkdir(exist_ok=True, parents=True)
        self._generate_file()
        self.script_file = self._modify_template()
        self.args.save_yaml(self.args.slurm_path.joinpath("slurm_conf.yaml"))

    def _launch(self):
        # ===== Submit the job =====
        print("Start to submit job!")
        process = subprocess.Popen(["sbatch", self.script_file], stdout=subprocess.PIPE, text=True)
        output, error = process.communicate()
        print("[script output]", output)
        print(
            "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
                self.script_file, self.args.slurm_path.joinpath(f"{self.args.job_name}.log")
            )
        )
        # sys.exit(0)

    def run(self):
        self._prepare_script()
        self._launch()

def slurm_launch(
    exp_name,
    num_nodes,
    num_gpus,
    partition,
    node_list="",
    load_env="",
    log_dir="logging",
    command_suffix="",
):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            conf = SlurmConfig(
                exp_name=exp_name,
                num_nodes=num_nodes,
                num_gpus=num_gpus,
                partition=partition,
                node_list=node_list,
                load_env=load_env,
                command_suffix=command_suffix,
                log_dir=log_dir,
            )
            deployment = SlurmDeploy(func, config=conf)
            return deployment.run()

        return wrapper

    return decorator
