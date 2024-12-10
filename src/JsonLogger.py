import json
import os

class JsonLogger:
    def __init__(self, max_entries=2, img_threshold=10):
        self.max_entries = int(max_entries)
        self.img_threshold = int(img_threshold)  # Пороговое значение для img

    def initialize_json_file(self, filename):
        """Проверка, существует ли файл, и создание его при отсутствии."""
        directory = os.path.dirname(filename)  # Получаем путь к директории файла
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)  # Создаем директории, если их нет
        if not os.path.exists(filename):
            with open(filename, 'w') as json_file:
                json.dump([], json_file, indent=4)  # Создаем пустой список

    def read_json(self, filename):
        """Чтение данных из JSON-файла."""
        try:
            with open(filename, 'r') as json_file:
                content = json_file.read()
                if content.strip() == "":
                    return []
                return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Ошибка при чтении JSON: {e}")
            return []
        except FileNotFoundError:
            print(f"Файл {filename} не найден.")
            return []
        except Exception as e:
            print(f"Произошла ошибка: {e}")
            return []

    def write_json(self, filename, data):
        """Запись данных в JSON-файл."""
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def update_img_values(self, data, img):
        """Обновление значений img в существующих записях."""
        for entry in data:
            if entry['img'] > 0 and img:
                entry['img'] += 1
            if entry['img'] > self.img_threshold:
                entry['img'] = -1
        return data

    def add(self, filename, user, text, img):
        """Добавление новой записи в JSON-файл."""
        # Инициализируем файл, если он не существует
        self.initialize_json_file(filename)

        # Сначала читаем существующие данные
        data = self.read_json(filename)

        # Обновляем значения img
        if img != -1:
            data = self.update_img_values(data, img)

        # Добавляем новую запись
        new_entry = {
            "user": user,
            "text": text,
            "img": img
        }

        data.append(new_entry)  # Добавляем новую запись в список

        # Проверяем, если количество записей превышает max_entries
        if len(data) > self.max_entries:
            # Удаляем старые записи (с начала списка)
            data = data[-self.max_entries:]  # Оставляем только последние max_entries записей

        # Записываем обновленные данные обратно в файл
        self.write_json(filename, data)

    def get(self, filename):
        """Получение всех записей из JSON-файла."""
        return self.read_json(filename)
