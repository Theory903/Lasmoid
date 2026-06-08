import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup a clean logger
logger = logging.getLogger("HFAnyUploader")
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[HF Uploader] %(levelname)s: %(message)s"))
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

class HFAnyUploader:
    def __init__(self, repo_id=None, token=None, max_workers=2):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.api = None
        
        if self.repo_id:
            try:
                from huggingface_hub import HfApi, create_repo
                self.api = HfApi(token=self.token)
                # Attempt to create the repository if it doesn't exist
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=True,
                    exist_ok=True
                )
                logger.info(f"Hugging Face repository '{self.repo_id}' is ready.")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Hugging Face Hub repo '{self.repo_id}': {e}. "
                    "Checkpoint uploading will be ignored/disabled, but training will continue normally."
                )
                self.api = None

    def upload_file_async(self, file_path, path_in_repo, commit_message=None):
        if not self.api or not self.repo_id:
            return
        
        file_path = str(file_path)
        logger.info(f"Queueing upload of file '{file_path}' to repository path '{path_in_repo}'...")
        
        def _upload():
            try:
                self.api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    commit_message=commit_message,
                    token=self.token,
                )
                logger.info(f"✅ Successfully uploaded file '{file_path}' -> '{path_in_repo}'")
            except Exception as e:
                logger.error(f"❌ Failed to upload file '{file_path}': {e}")

        future = self.executor.submit(_upload)
        self.futures.append(future)

    def upload_folder_async(self, folder_path, path_in_repo, commit_message=None):
        if not self.api or not self.repo_id:
            return
        
        folder_path = str(folder_path)
        logger.info(f"Queueing upload of folder '{folder_path}' to repository path '{path_in_repo}'...")
        
        def _upload():
            try:
                self.api.upload_folder(
                    folder_path=folder_path,
                    path_in_repo=path_in_repo,
                    repo_id=self.repo_id,
                    commit_message=commit_message,
                    token=self.token,
                )
                logger.info(f"✅ Successfully uploaded folder '{folder_path}' -> '{path_in_repo}'")
            except Exception as e:
                logger.error(f"❌ Failed to upload folder '{folder_path}': {e}")

        future = self.executor.submit(_upload)
        self.futures.append(future)

    def wait_for_uploads(self):
        if not self.futures:
            return
        logger.info("Waiting for all pending Hugging Face uploads to finish...")
        for future in as_completed(self.futures):
            pass
        self.futures.clear()
        logger.info("All pending Hugging Face uploads completed.")
