계속 접속 유지하려면
ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 user@hostname


혹은 tmux를 쓰도록 하자