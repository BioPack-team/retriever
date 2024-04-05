import { BioLink } from 'biolink-model';
import { BioLinkHandlerClass } from './common/types';

class BioLinkHandler implements BioLinkHandlerClass {
  private static instance: BioLinkHandler;
  private _classTree: BioLink['classTree'];

  constructor() {
    if (!BioLinkHandler.instance) {
      const biolink = new BioLink();
      biolink.loadSync();
      this._classTree = biolink.classTree;
    }
    return BioLinkHandler.instance;
  }

  get classTree() {
    return this._classTree;
  }
}

const BioLinkHandlerInstance = new BioLinkHandler();
Object.freeze(BioLinkHandlerInstance);

export default BioLinkHandlerInstance;
