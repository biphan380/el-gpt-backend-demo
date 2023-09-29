from llama_index import SimpleDirectoryReader
from typing import Callable, Dict, Generator, List, Optional, Type
from llama_index.schema import Document
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS

class CustomDirectoryReader(SimpleDirectoryReader):
    
    def load_data(self) -> List[Document]:
        """We attempt to Load data from the input directory.
        We want each pdf to be its own Document object, not be 
        split up into N Documents objects for N pages.
        
        Returns:
            List[Document]: A list of Document objects. But each file is 
            its own Document."""
        
        documents = []
        for input_file in self.input_files:
            metadata: Optional[dict] = None
            if self.file_metadata is not None:
                metadata = self.file_metadata(str(input_file))

            # Add the title of the file to the metadata
            title = input_file.stem  # This gives the file name without the extension
            if metadata is None:
                metadata = {'title': title}
            else:
                metadata.update({'title': title})

            file_suffix = input_file.suffix.lower()
            if (file_suffix in self.supported_suffix or file_suffix in self.file_extractor):
                # use file readers
                if file_suffix not in self.file_extractor:
                    # instantiate file reader if not only
                    reader_cls = DEFAULT_FILE_READER_CLS[file_suffix]
                    self.file_extractor[file_suffix] = reader_cls()

                reader = self.file_extractor[file_suffix]
                docs = reader.load_data(input_file, extra_info=metadata)

                # Combine the texts of all the documents into one single document
                combined_text = ' '.join(doc.text for doc in docs)
                combined_document = Document(text=combined_text, metadata=metadata or {})
                if self.filename_as_id:
                    combined_document.id_ = str(input_file)

                documents.append(combined_document)
            else:
                # do standard read
                with open(input_file, "r", errors=self.errors, encoding=self.encoding) as f:
                    data = f.read()

                doc = Document(text=data, metadata=metadata or {})
                if self.filename_as_id:
                    doc.id_ = str(input_file)

                documents.append(doc)

        return documents
