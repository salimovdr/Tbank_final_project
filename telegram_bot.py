# 7619446461:AAEtxfJ7AIpsuFRlI6nUgtNO4F6PWev2uNM



from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import logging
import os

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальные переменные
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
N_MAX_PHOTO = 3  # Максимальное количество фото на пользователя

# Генерация клавиатуры
def generate_keyboard():
    return ReplyKeyboardMarkup(
        [["Последнее"], ["Загрузить", "Сгенерировать"]],
        resize_keyboard=True
    )

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    logger.info(f"Пользователь {chat_id} запустил бота.")
    await update.message.reply_text(
        "Добро пожаловать! Выберите действие:",
        reply_markup=generate_keyboard()
    )

# Выбор 1го приоритета
def manage_photo_storage(user_dir):
    files = sorted([f for f in os.listdir(user_dir) if f.isdigit()], key=int)

    # Переименование файлов
    for i, file_name in enumerate(files, start=1):
        old_name = os.path.join(user_dir, file_name)
        new_name = os.path.join(user_dir, str(i))
        if old_name != new_name:
            os.rename(old_name, new_name)
            logger.info(f"Переименовано {old_name} -> {new_name}")


# Переименование файлов
def manage_photo_storage(user_dir):
    files = sorted([f for f in os.listdir(user_dir) if f.isdigit()], key=int, reverse=True)
    
    # Переименование файлов
    for i in range(len(files), 0, -1):
        old_name = os.path.join(user_dir, str(i))
        new_name = os.path.join(user_dir, str(i + 1))
        os.rename(old_name, new_name)
        logger.info(f"Переименовано {old_name} -> {new_name}")

    files = sorted([f for f in os.listdir(user_dir) if f.isdigit()], key=int, reverse=True)
    # print(files)

    # Удаление самого старого файла, если превышено N_MAX_PHOTO
    if len(files) >= N_MAX_PHOTO:
        os.remove(os.path.join(user_dir, files[0]))
        logger.info(f"Удалено старое фото в директории {user_dir} {files[1]}.")

# Обработка изображений
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]  # Берем изображение в наибольшем разрешении

    # Создание директории для пользователя
    user_dir = os.path.join(UPLOAD_DIR, str(chat_id))
    os.makedirs(user_dir, exist_ok=True)

    # Управление хранилищем фото
    manage_photo_storage(user_dir)

    # Сохранение нового фото под именем "1"
    file_path = os.path.join(user_dir, "1")
    try:
        file = await photo.get_file()
        await file.download_to_drive(file_path)  # Сохранение изображения на диск
        logger.info(f"Новое изображение от {chat_id} сохранено как {file_path}.")
        await update.message.reply_text("Файл успешно загружен, обрабатываем...")
    except Exception as e:
        logger.error(f"Ошибка при сохранении изображения от {chat_id}: {e}")
        await update.message.reply_text("Произошла ошибка при сохранении файла. Попробуйте снова.")


# Обработка кнопки "Последнее"
async def handle_last(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_dir = os.path.join(UPLOAD_DIR, str(chat_id))
    
    if os.path.exists(user_dir) and os.listdir(user_dir):
        files = sorted([f for f in os.listdir(user_dir) if f.isdigit()], key=int)
        for file_name in files:
            file_path = os.path.join(user_dir, file_name)
            with open(file_path, "rb") as file:
                await update.message.reply_photo(photo=file, caption=f"Фото {file_name}")
        
        # Сохранение состояния пользователя для ожидания ввода
        context.user_data["awaiting_photo_selection"] = True
        await update.message.reply_text(
            f"Выберите изображение, с которым хотите работать, введя его номер (1 - {N_MAX_PHOTO}).",
            reply_markup=ReplyKeyboardRemove()  # Удаление клавиатуры
        )
        logger.info(f"Последние файлы отправлены пользователю {chat_id}.")
    else:
        await update.message.reply_text("Файлы не найдены. Загрузите файлы с помощью кнопки 'Загрузить'.")
        logger.info(f"Пользователь {chat_id} запросил последние файлы, но файлов нет.")



# Обработка кнопки "Сгенерировать"
async def handle_generate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text("Генерируем...")
    logger.info(f"Пользователь {chat_id} запросил генерацию.")
    
    # Отправка изображения (замените 'img.jpeg' на нужный путь)
    try:
        with open("img.jpeg", "rb") as img:
            await update.message.reply_photo(photo=img, caption="Вот сгенерированное изображение!")
    except FileNotFoundError:
        logger.error("Файл img.jpeg не найден!")
        await update.message.reply_text("Ошибка: файл для генерации отсутствует.")


async def handle_photo_selection(chat_id: int, selected_number: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_dir = os.path.join(UPLOAD_DIR, str(chat_id))
    selected_file = os.path.join(user_dir, str(selected_number))
    first_file = os.path.join(user_dir, "1")

    if os.path.exists(selected_file) and os.path.exists(first_file):
        temp_file = os.path.join(user_dir, "temp")
        os.rename(first_file, temp_file)
        os.rename(selected_file, first_file)
        os.rename(temp_file, selected_file)
        # Отправка сообщения с клавиатурой
        await context.bot.send_message(
            chat_id,
            f"Фото {selected_number} и 1 поменялись местами.",
            reply_markup=ReplyKeyboardMarkup(
                [["Последнее"], ["Загрузить", "Сгенерировать"]],
                resize_keyboard=True
            )
        )
        logger.info(f"Фото {selected_number} и 1 поменялись местами для пользователя {chat_id}.")
    else:
        await context.bot.send_message(chat_id, "Выбранное фото или фото 1 не существует.")




# Обработка текстовых сообщений (клавиши и выбор фото)
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    chat_id = update.message.chat_id

    # Проверка, ждет ли бот выбора фотографии
    if context.user_data.get("awaiting_photo_selection"):
        try:
            selected_number = int(text)
            if 1 <= selected_number <= N_MAX_PHOTO:
                await handle_photo_selection(chat_id, selected_number, context)
                context.user_data["awaiting_photo_selection"] = False  # Сброс состояния
            else:
                await update.message.reply_text(f"Введите корректный номер от 1 до {N_MAX_PHOTO}.")
        except ValueError:
            await update.message.reply_text("Пожалуйста, введите число.")
        return

    # Стандартные команды
    if text == "Загрузить":
        await update.message.reply_text("Пожалуйста, отправьте файл (изображение) для загрузки.")
        logger.info(f"Пользователь {chat_id} нажал 'Загрузить'.")
    elif text == "Сгенерировать":
        await handle_generate(update, context)
    elif text == "Последнее":
        await handle_last(update, context)
    else:
        await update.message.reply_text("Неизвестная команда. Пожалуйста, используйте кнопки.")
        logger.warning(f"Пользователь {chat_id} отправил неизвестную команду: {text}.")



# Основная функция
def main():
    # Создание бота
    application = Application.builder().token("7619446461:AAEtxfJ7AIpsuFRlI6nUgtNO4F6PWev2uNM").build()

    # Обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT, text_handler))

    # Запуск бота
    logger.info("Бот запущен.")
    application.run_polling()

if __name__ == "__main__":
    main()




