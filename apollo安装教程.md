# apollo 安装教程


## 系统要求
主机系统必须是ubuntu14.04

## 安装docker-ce
卸载老版本的docker（如果之前未安装docker，忽略此步骤）  

    $sudo apt-get remove docker docker-engine docker.io
按顺序执行下列命令 

    $ sudo apt-get update

    $ sudo apt-get install \
        linux-image-extra-$(uname -r) \
        linux-image-extra-virtual

    $ sudo apt-get update

    $ sudo apt-get install \
        apt-transport-https \
        ca-certificates \
        curl \
        software-properties-common

    $ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

    $ sudo apt-key fingerprint 0EBFCD88

    $ sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"
    $ sudo apt-get update

    $ sudo apt-get install docker-ce

让docker能够以普通权限运行

    $ sudo groupadd docker

    $ sudo usermod -aG docker $USER

测试docker是否正安装及设置（如果没有报错，说明安装成功）

    $ docker run hello-world

更换源(如果不换源，docker拉取镜像的速度会灰常灰常慢)

    $ sudo gedit /etc/docker/daemon.json
    在打开的文件中输入：  
    {
         "registry-mirrors": ["http://hub-mirror.c.163.com"]
    }

重启电脑

#  安装apollo
从github下载apollo源码，https://github.com/ApolloAuto/apollo  

下载的压缩包名为“apollo-master.zip”。解压，并进入apollo-master目录。  

    cd apollo-master

下载并启动apollo release版本：(输入下面的命令后，会提示输入‘y’，按要求输入y即可)

    bash docker/scripts/release_start.sh

在浏览器中打开 http://localhost:8888/ (注意，不要使用代理！！！),显示如下

![](https://github.com/RockAutonomous/blog/blob/master/images/Dreamview1.png)

进入apollo（如果想在多个终端中进入apollo，均是输入下面的命令）：

    bash docker/scripts/release_into.sh

循环播放rosbag：

    rosbag play docs/demo_guide/demo.bag -l 

在浏览器中，将会出现运动的小轿车，类似如下情景：

![](https://github.com/RockAutonomous/blog/blob/master/images/Dreamview2.png)

end
