import cv2
import os
import telebot

from selfie_segmentation import segment_an_image, patch_hair, patch_background

bot = telebot.TeleBot(os.getenv('TELEGRAM_BOT_TOKEN_ENV'))


def __get_chat_root_folder(message) -> str:
    chat_root_folder = os.path.join(".", str(message.chat.id))
    if not os.path.exists(chat_root_folder):
        os.makedirs(chat_root_folder, exist_ok=True)
    return chat_root_folder


def handler_segmentation(message):
    if message.content_type != "photo":
        bot.send_message(message.chat.id, f"Compressed image expected, but got {message.content_type}.")
        return

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    chat_root_folder = __get_chat_root_folder(message)

    downloaded_file_name = os.path.join(chat_root_folder, "raw_image.jpg")
    with open(downloaded_file_name, "wb") as f:
        f.write(downloaded_file)

    bot.send_message(message.chat.id, "Performing segmentation.")

    res = segment_an_image(downloaded_file_name)

    reply_message = "The images below are:\n"
    media = []
    for i, image_data in enumerate(res["images"]):
        image, name = image_data
        reply_message += f"\t{i:02d}. {name}\n"
        full_path = os.path.join(".", chat_root_folder, f"{name}.png")
        cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        media.append(telebot.types.InputMediaPhoto(media=open(full_path, "rb"), caption=name))

    bot.send_message(message.chat.id, res["info"])
    bot.send_message(message.chat.id, reply_message)
    bot.send_media_group(chat_id=message.chat.id, media=media)


def handler_change_hair_color_step_one(message):
    if message.content_type != "photo":
        bot.send_message(message.chat.id, f"Compressed image expected, got {message.content_type}. Aborting...")
        return

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    chat_root_folder = __get_chat_root_folder(message)

    downloaded_file_name = os.path.join(chat_root_folder, "portrait.jpg")
    with open(downloaded_file_name, "wb") as f:
        f.write(downloaded_file)

    send = bot.send_message(message.chat.id, "Okay, now let's send a new color.")
    bot.register_next_step_handler(send, lambda m: handler_change_hair_color_step_two(m, downloaded_file_name))


def handler_change_hair_color_step_two(message, portrait_file_name):
    def hex_to_rgb(hexa: str) -> tuple[int, int, int]:
        return int(hexa[1:3], 16), int(hexa[3:5], 16), int(hexa[5:7], 16)

    if message.content_type != "text":
        bot.send_message(message.chat.id, f"Text was expected, got {message.content_type}. Aborting...")
        return

    if not os.path.exists(portrait_file_name):
        bot.send_message(message.chat.id, "Something went wrong...")
        return

    chat_root_folder = __get_chat_root_folder(message)

    text = message.text
    color = hex_to_rgb(text)

    res = patch_hair(portrait_file_name, color)
    image = res['image']

    full_path = os.path.join(".", chat_root_folder, "dyed_hairs.png")
    cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    media = [telebot.types.InputMediaPhoto(media=open(full_path, "rb"), caption="dyed hairs")]

    bot.send_message(message.chat.id, res["info"])
    bot.send_media_group(chat_id=message.chat.id, media=media)


def handler_change_background_step_one(message):
    if message.content_type != "photo":
        bot.send_message(message.chat.id, f"Compressed image expected, got {message.content_type}. Aborting...")
        return

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    chat_root_folder = __get_chat_root_folder(message)

    downloaded_file_name = os.path.join(chat_root_folder, "portrait.jpg")
    with open(downloaded_file_name, "wb") as f:
        f.write(downloaded_file)

    send = bot.send_message(message.chat.id, "Okay, now let's send a new background.")
    bot.register_next_step_handler(send, lambda m: handler_change_background_step_two(m, downloaded_file_name))


def handler_change_background_step_two(message, portrait_file_name):
    if message.content_type != "photo":
        bot.send_message(message.chat.id, f"Compressed image expected, got {message.content_type}. Aborting...")
        return

    if not os.path.exists(portrait_file_name):
        bot.send_message(message.chat.id, "Something went wrong...")
        return

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    background_file = bot.download_file(file_info.file_path)

    chat_root_folder = __get_chat_root_folder(message)

    background_file_name = os.path.join(chat_root_folder, "background.jpg")
    with open(background_file_name, "wb") as f:
        f.write(background_file)

    res = patch_background(background_image_path=background_file_name,
                           foreground_image_path=portrait_file_name)
    image = res['image']

    full_path = os.path.join(".", chat_root_folder, "patched_background.png")
    cv2.imwrite(full_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    media = [telebot.types.InputMediaPhoto(media=open(full_path, "rb"), caption="patched background")]

    bot.send_message(message.chat.id, res["info"])
    bot.send_media_group(chat_id=message.chat.id, media=media)


@bot.message_handler(commands=["segmentation"])
def command_segmentation(message):
    assert message is not None
    send = bot.send_message(message.chat.id, "Send a portrait to start.")
    bot.register_next_step_handler(send, handler_segmentation)


@bot.message_handler(commands=["change-hair-color"])
def change_hair_color(message):
    assert message is not None
    send = bot.send_message(message.chat.id, "Send a portrait to start.")
    bot.register_next_step_handler(send, handler_change_hair_color_step_one)


@bot.message_handler(commands=["change-background"])
def change_background(message):
    assert message is not None
    send = bot.send_message(message.chat.id, "Send a portrait to start.")
    bot.register_next_step_handler(send, handler_change_background_step_one)


bot.polling(none_stop=True, interval=0)
