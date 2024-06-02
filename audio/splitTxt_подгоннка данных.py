import os


def split_txt_file(input_file_path):
    # Чтение содержимого исходного файла
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Инициализация переменных
    i = 0
    file_count = 1

    while i < len(lines):
        # Получение количества строк для текущего блока
        count = int(lines[i].strip())
        i += 1

        # Получение строк для текущего блока
        block_lines = lines[i:i + count]
        i += count

        # Сохранение блока в новый файл
        output_file_path = f"{input_file_path.rsplit('.', 1)[0]}_{file_count}.txt"
        with open(output_file_path, 'w') as output_file:
            output_file.write(f"{count}\n")
            output_file.writelines(block_lines)

        file_count += 1

    # Удаление исходного файла
    os.remove(input_file_path)


# Укажите путь к исходному файлу
input_file_path = r'C:\Users\admin\Desktop\hackaton 2024\test_recognize_people\yovol8\dataset_bacaptxt\train\not\1.txt'

# Вызов функции
split_txt_file(input_file_path)