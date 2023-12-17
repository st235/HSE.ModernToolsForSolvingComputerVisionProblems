FROM python:3.6

RUN mkdir -p /usr/src/telegram_bot/
WORKDIR /usr/src/telegram_bot/

COPY ./requirements.txt /usr/src/telegram_bot/
COPY ./telegram_bot.py /usr/src/telegram_bot/
COPY ./sobel.py /usr/src/telegram_bot/
COPY ./package.tar /usr/src/telegram_bot/

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "telegram_bot.py"]