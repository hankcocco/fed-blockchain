## build镜像
docker build -t blockchain:latest .

## 启动镜像 
docker-compose up -d

docker run -d -p 3009:3009 --name blockchain blockchain 

## 查看容器日志
sudo docker logs -f --details blockchain1