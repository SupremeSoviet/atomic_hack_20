import os
import asyncio
import cv2
from telebot import types
from telebot.async_telebot import AsyncTeleBot
from ultralytics import YOLO
import nest_asyncio

bot = AsyncTeleBot('key')

TEMP_DIR = '/content/temp_files/'

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

model = YOLO("best (2).pt")

@bot.message_handler(commands=['start'])
async def send_welcome(message):
    await bot.reply_to(message, "Привет, жду от тебя видео или фото")

@bot.message_handler(content_types=['video'])
async def handle_video(message):
    video = message.video
    file_info = await bot.get_file(video.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)

    input_path = os.path.join(TEMP_DIR, video.file_id + '.mp4')
    with open(input_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    output_path = os.path.join(TEMP_DIR, 'output_' + video.file_id + '.mp4')

    await process_video(input_path, output_path)
    await bot.send_video(message.chat.id, open(output_path, 'rb'))

    os.remove(input_path)
    os.remove(output_path)

@bot.message_handler(content_types=['photo'])
async def handle_photo(message):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    downloaded_file = await bot.download_file(file_info.file_path)

    input_path = os.path.join(TEMP_DIR, photo.file_id + '.jpg')
    with open(input_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    output_path = os.path.join(TEMP_DIR, 'output_' + photo.file_id + '.jpg')

    await process_photo(input_path, output_path)
    await bot.send_photo(message.chat.id, open(output_path, 'rb'))

    os.remove(input_path)
    os.remove(output_path)

async def process_photo(input_path, output_path):
    image = cv2.imread(input_path)

    results = model(image)

    annotated_image = image.copy()
    for r in results:
        annotated_image = r.plot()

    cv2.imwrite(output_path, annotated_image)

async def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            annotated_frame = r.plot()
            out.write(annotated_frame)

        await asyncio.sleep(0)

    cap.release()
    out.release()

async def main():
    await bot.polling()

if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())