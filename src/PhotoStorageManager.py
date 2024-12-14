import os


class PhotoStorageManager:
    """Manages photo storage for each user."""
    @staticmethod
    def ensure_directory_exists(directory):
        os.makedirs(directory, exist_ok=True)

    @staticmethod
    def manage_photos(user_dir, max_photos):
        files = os.listdir(user_dir)

        if len(files) >= int(max_photos):
            os.remove(os.path.join(user_dir, files[0]))
            for i in range(1, len(files)):
                old_name = os.path.join(user_dir, str(i))
                new_name = os.path.join(user_dir, str(i - 1))
                if os.path.exists(old_name):
                    os.rename(old_name, new_name)
                    
