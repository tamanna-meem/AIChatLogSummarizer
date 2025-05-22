import os

def main():
    chat_file_path = 'sample_chats/chat.txt'
    output_file_path = 'output/raw_chat.txt'

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            chat_content = f.read()
    except FileNotFoundError:
        print(f"Error: Chat file not found at {chat_file_path}")
        return

    print(chat_content)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(chat_content)
        print(f"Chat saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_file_path}")

if __name__ == "__main__":
    main()
