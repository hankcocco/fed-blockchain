
FROM python:3.7

# 修改时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# 设置编码
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
# change pip resource
COPY pip.conf /root/.pip/pip.conf
# 拷贝当前目录下的requirements.txt到容器中
COPY requirements.txt /requirements.txt

# 拷贝torch文件到容器中
COPY torch-1.7.1-cp37-cp37m-manylinux1_x86_64.whl /torch-1.7.1-cp37-cp37m-manylinux1_x86_64.whl

# 安装pip包
RUN pip install /torch-1.7.1-cp37-cp37m-manylinux1_x86_64.whl  \
    && pip install -r /requirements.txt

# 设置项目变量名
ENV PROJECT_DIR=/app

# 拷贝当前所有文件到/app目录下
COPY . ${PROJECT_DIR}
# 进入到/app目录下
WORKDIR ${PROJECT_DIR}

# 执行 启动命令
# CMD python console.py node run 3009 start 1000 end 2000