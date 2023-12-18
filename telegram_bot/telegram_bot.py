import cv2
import os
import telebot

from selfie_segmentation import segment_an_image

print(os.getenv('TELEGRAM_BOT_TOKEN'))
bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN'))


def handler_sobel(message):
    if message.content_type == "photo":
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        downloaded_file_path = os.path.join(".", str(message.chat.id))
        if not os.path.exists(downloaded_file_path):
            os.makedirs(downloaded_file_path)

        downloaded_file_name = os.path.join(downloaded_file_path, "image_to_sobel.jpg")
        with open(downloaded_file_name, "wb") as f:
            f.write(downloaded_file)

        bot.send_message(message.chat.id, "Starting...")

        res = segment_an_image(downloaded_file_name)

        reply = ""
        media = []
        for i, image_data in enumerate(res["images"]):
            name, image = image_data
            reply += "{}. {}\n".format(i, name)
            full_path = os.path.join(".", str(message.chat.id), name)
            cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            media.append(
                telebot.types.InputMediaPhoto(media=open(full_path, "rb"), caption=name)
            )

        bot.send_message(message.chat.id, res["info"])
        bot.send_message(message.chat.id, reply)
        bot.send_media_group(chat_id=message.chat.id, media=media)
    else:
        bot.send_message(
            message.chat.id,
            "Something was wrong. Please try again and send one compressed image.",
        )


@bot.message_handler(commands=["sobel"])
def command_sobel(message):
    if message:
        send = bot.send_message(message.chat.id, "Send an image:")
        bot.register_next_step_handler(send, handler_sobel)
    else:
        bot.send_message(message.chat.id, "Something was wrong. Please try again.")


bot.polling(none_stop=True, interval=0)
