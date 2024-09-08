import yaml
import re
 
# バージョン指定の有無
version = True
# yamlファイル
yaml_path = "environment.yml"
 
with open(yaml_path) as data:
    yaml_obj = yaml.safe_load(data)
 
    requirements = []
    for dep in yaml_obj['dependencies']:
        if isinstance(dep, str):
            dep_l = re.split('=', dep)
            # 除外対象
            res = re.match('python|pip|setuptools', dep)
            if res is None:
                if version and len(dep_l) == 2:
                    requirements.append(dep_l[0] + '==' + dep_l[1])
                else:
                    requirements.append(dep_l[0])
        else:
            for preq in dep.get('pip', []):
                preq_s = re.sub('>=|<=|>|<|==', '#', preq)
                preq_s_l = re.split('#', preq_s)
 
                if preq_s_l[0]:
                    res = re.match('-e', preq_s_l[0])
                    if res is None:
                        new_string = preq_s_l[0]
                    else:
                        new_string = preq.lstrip("-e | -e .")
                        new_string = new_string.strip()
 
                if version:
                    new_string = preq.lstrip("-e | -e .")
                    requirements.append(new_string)
                else:
                    requirements.append(new_string)
 
with open('requirements.txt', 'w') as fp:
    for requirement in requirements:
        print(requirement, file=fp)