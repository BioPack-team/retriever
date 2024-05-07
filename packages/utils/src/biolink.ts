import { BioLink } from '@retriever/biolink-model';
import Debug from 'debug';
const debug = Debug('retriever:EdgeReverse');

class BioLinkModel {
  biolink: BioLink;
  constructor() {
    this.biolink = new BioLink();
    this.biolink.loadSync();
  }

  reverse(predicate: string) {
    if (typeof predicate === 'string') {
      if (predicate in this.biolink.slotTree.objects) {
        if (this.biolink.slotTree.objects[predicate].symmetric === true) {
          return predicate;
        }
        return this.biolink.slotTree.objects[predicate].inverse;
      }
    }

    return undefined;
  }

  getAncestorClasses(className: string): string | string[] {
    if (className in this.biolink.classTree.objects) {
      const ancestors = this.biolink.classTree.getAncestors(className).map((entity) => entity.name);
      return [...ancestors, ...[className]];
    }
    return className;
  }

  getAncestorPredicates(predicate: string): string | string[] {
    if (predicate in this.biolink.slotTree.objects) {
      const ancestors = this.biolink.slotTree.getAncestors(predicate).map((entity) => entity.name);
      return [...ancestors, ...[predicate]];
    }
    return predicate;
  }

  getDescendantClasses(className: string): string | string[] {
    if (className in this.biolink.classTree.objects) {
      const descendants = this.biolink.classTree.getDescendants(className).map((entity) => entity.name);
      return [...descendants, ...[className]];
    }
    return className;
  }

  getDescendantPredicates(predicate: string): string[] {
    if (predicate in this.biolink.slotTree.objects) {
      const descendants = this.biolink.slotTree.getDescendants(predicate).map((entity) => entity.name);
      return [...descendants, ...[predicate]];
    }
    return [predicate];
  }

  getDescendantQualifiers(qualifier: string): string[] {
    try {
      const descendants = this.biolink.enumTree.getDescendants(qualifier).map((entity) => entity.name);
      return [...descendants, qualifier];
    } catch (e) {
      console.log('qual error', e);
      return [qualifier];
    }
  }
}

// Freeze an instance to avoid multiple reloads
const biolink = new BioLinkModel();
Object.freeze(biolink);

global.BIOLINK_VERSION = biolink.biolink.biolinkJSON.version;

export { biolink };
