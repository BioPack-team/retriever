import { BioLinkClass, BioLinkEntityObject } from '../types/types';
import { pascalCase } from 'pascal-case';
import BaseObject from './base_object';

export default class Entity extends BaseObject implements BioLinkEntityObject {
  private _id_prefixes: string[];

  constructor(name: string, info: BioLinkClass) {
    super(name, info);
    this.id_prefixes = info.id_prefixes;
    this._parent = typeof info.is_a === 'undefined' ? info.is_a : pascalCase(info.is_a);
  }

  get id_prefixes(): string[] {
    return this._id_prefixes;
  }

  set id_prefixes(idPrefixes: string[]) {
    if (this._name === 'Gene') {
      this._id_prefixes = [...['SYMBOL', 'OMIM', 'UMLS'], ...idPrefixes];
    } else if (this._name === 'ChemicalSubstance') {
      this._id_prefixes = typeof idPrefixes === 'undefined' ? ['UMLS'] : [...['UMLS'], ...idPrefixes];
    } else if (this._name === 'SmallMolecule') {
      this._id_prefixes = [...['UMLS'], ...idPrefixes];
    } else if (this._name === 'Disease') {
      this._id_prefixes = [...['GARD'], ...idPrefixes];
    } else {
      this._id_prefixes = idPrefixes;
    }
  }

  addChild(child: string): void {
    this._children.push(pascalCase(child));
  }
}
