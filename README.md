# 参赛须知

- 下载Playground环境以及配置对应虚拟环境

	- 下载环境以及example

	```bash
	git clone git@github.com:mail-ecnu/example_playground.git
	```

	- 配置环境：

		- Note：若您使用了额外的python包，请在requirements.txt添加附加依赖。
		- 在example_playground文件夹，配置pommerman

		```bash
		cd example_playground
		conda env create -f env.yml
		conda activate pommerman
		```

- 自定义Docker-Agent

	- 参考[main分支](https://github.com/mail-ecnu/example_playground)中example_playground/examples/docker-agent下的train.py，run.py两个文件或者参考[A2C分支](https://github.com/mail-ecnu/example_playground/tree/A2C)下example_playground/examples/docker-agent/A2C/main.py文件完成你的算法。（两个实例代码都包含了完整的训练代码）
	- 你的Agent类需要继承agents.BaseAgent，必须要完善的是agent的act方法，在这里完善你的policy。
	- 完善你的算法，使用您的算法，训练和保存模型；

- 测试您的算法

	- 提交代码前可以进行本地测试保证你的代码可以成功运行。

	- 根据dockerfile生成镜像。

		```bash
		sudo docker build -t pommerman/simple-agent -f examples/docker-agent/Dockerfile .
		```

	- 运行比赛

		```
		sudo python examples/simple_ffa_run.py
		```

		您可以通过修改examples/simple_ffa_run.py中的参数测试您的算法性能。

		如果docker运行失败需要多次运行以上代码。

- 提交您的代码

	- 在校园网环境下访问比赛网页（内部测试阶段）注册账户，在网页的profile中上传你的代码对应的github私钥，然后进行提交。
	- 提交页面需要输入三项信息：AgentName，Your repository url，Your dockerfile path，需要注意url必须选择ssh格式。
