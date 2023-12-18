FROM python:3.9-slim-bullseye

RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 -y

RUN mkdir -p /usr/src/telegram_bot/
WORKDIR /usr/src/telegram_bot/

COPY ./requirements.txt /usr/src/telegram_bot/
COPY ./telegram_bot.py /usr/src/telegram_bot/
COPY ./selfie_segmentation.py /usr/src/telegram_bot/
COPY ./utils.py /usr/src/telegram_bot/
COPY ./package.tar /usr/src/telegram_bot/
COPY ./parameters.npy /usr/src/telegram_bot/

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "telegram_bot.py"]