# base image
FROM python:3.8

WORKDIR /app
RUN useradd appuser && chown -R appuser /app

# streamlit-specific commands
RUN mkdir -p /root/.streamlit

# copy over and install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt


RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
serverAddress = 0.0.0.0\n\
" > /root/.streamlit/config.toml'
# copying everything over

#COPY . .
# exposing default port for streamlit
EXPOSE 8501
EXPOSE 8502

USER appuser
# run app
#CMD streamlit run #webapp/app.py

