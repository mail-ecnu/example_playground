# 参赛须知

* 自定义Docker-Agent算法
  * 使用您的算法，完善pommerman/runner/docker_agent_runner.py中的 act() 函数；
  * Note：若您使用了额外的python包，请在requirements.txt添加附加依赖。

* 测试您的算法

  * 进入下载的playground文件夹，配置pommerman

    ```bash
    cd playground
    conda env create -f env.yml
    conda activate pommerman
    ```

  * 安装对应docker镜像中的agent。这个安装过程比较长，需要下载比较多东西。

    ```bash
    docker build -t pommerman/simple-agent -f examples/docker-agent/Dockerfile .
    ```

  * 运行比赛

    ```
    python examples/simple_ffa_run.py
    ```

    您可以通过修改和输出examples/simple_ffa_run.py中的参数测试您的算法性能。

