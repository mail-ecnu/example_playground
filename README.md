# 参赛须知

* 自定义Docker-Agent算法
  * 使用您的算法，修改 examples/docker-agent/train.py 函数并且训练和保存模型；
  * 使用您的模型与算法，完善run.py中Agent类的act函数
  * Note：若您使用了额外的python包，请在requirements.txt添加附加依赖。

* 测试您的算法

  * 进入下载的playground文件夹，配置pommerman

    ```bash
    cd playground
    conda env create -f env.yml
    conda activate pommerman
    ```
  
  * Note：训练前请完成以上步骤

  * 安装对应docker镜像中的agent。这个安装过程比较长，需要下载比较多东西。

    ```bash
    docker build -t pommerman/simple-agent -f examples/docker-agent/Dockerfile .
    ```

  * 运行比赛

    ```
    python examples/simple_ffa_run.py
    ```

    您可以通过修改和输出examples/simple_ffa_run.py中的参数测试您的算法性能。

