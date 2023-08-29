workers = 5    # 定义同时开启的处理请求的进程数量，根据网站流量适当调整
worker_class = "gevent"   # 采用gevent库，支持异步处理请求，提高吞吐量
bind = "0.0.0.0:3490"  # 监听IP和端口
daemon = False   # 是否后台运行
name = "gunicorn.pid"   # gunicorn进程名
