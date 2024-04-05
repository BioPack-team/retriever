export default class FileLoadingError extends Error {
  constructor(message = 'Failed to load the BioLink Model file.') {
    super();
    this.name = 'FileLoadingError';
    this.message = message;
  }
}
