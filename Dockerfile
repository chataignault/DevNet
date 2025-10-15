FROM python:3.9-slim

# git
RUN apt-get update && apt-get install -y git openssh-client
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh
COPY id_rsa /root/.ssh/
RUN chmod 600 /root/.ssh/id_rsa
RUN git config --global user.name "Your Name" && \
    git config --global user.email "your.email@example.com"
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# python
RUN pip install scikit-learn pandas numpy joblib
WORKDIR /packages  
RUN git clone <repository_url> .
RUN pip install -e DevNet

WORKDIR /workdir

# code
COPY data/ ./data/
COPY development/ ./development/
COPY models/ ./models/
COPY grid_search.py .

# run
CMD ["python", "grid_search.py"]

# Copy results back to host
CMD python grid_search.py 
CMD scp -i /root/.ssh/id_rsa /workdir/logs/grid_search_results.csv user@host:/path/to/destination/




