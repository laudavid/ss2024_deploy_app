FROM public.ecr.aws/lambda/python:3.8

# ARG email="example@ip-paris.fr"
# LABEL "maintainer"=$email
# LABEL "rating"="Five Stars" "class"="First Class"

# ENV AP /data/app
# ENV SCPATH /etc/supervisor/conf.d

# Install the function's dependencies using file requirements.txt
# from your project folder.
COPY saved_models ./saved_models
COPY nltkdata ./nltkdata


COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# RUN python -m nltk.downloader stopwords wordnet --target "${LAMBDA_TASK_ROOT}"

# Copy function code into LAMBDA Function Code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "app.handler" ]