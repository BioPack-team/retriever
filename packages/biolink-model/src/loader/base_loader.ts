import yaml from 'js-yaml';
import FileLoadingError from '../exceptions/file_loading_error';
import { BioLinkJSON, LoaderClass } from '../types/types';

export default abstract class Loader implements LoaderClass {
  protected yaml2json(input: string) {
    try {
      const doc = yaml.load(input);
      return doc;
    } catch (e) {
      throw new FileLoadingError('Failed to load BioLink Model. Unable to convert the yaml file into json.');
    }
  }
  abstract load(input: string | undefined): Promise<BioLinkJSON> | BioLinkJSON;
}
