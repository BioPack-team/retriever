import Slot from '../object/slot_object';
import { underscore } from '../utils';
import {
  BioLinkEnums,
  BioLinkQualifiersObject,
  BioLinkEnumTreeObject,
  BioLinkQualifierObject,
  BioLinkQualifiers,
  BioLinkQualifier,
} from '../types/types';
import BaseTree from './base_tree';
import Qualifier from '../object/qualifier_object';

export default class BioLinkEnumTree extends BaseTree implements BioLinkEnumTreeObject {
  protected _objects_in_yaml: BioLinkEnums;
  protected _objects_in_tree: BioLinkQualifiersObject;

  constructor(objects: BioLinkEnums) {
    let qualifier_objects: BioLinkQualifiers = {};
    Object.values(objects).forEach((enum_value) => {
      Object.keys(enum_value.permissible_values ?? {}).forEach((qualifier) => {
        qualifier_objects[qualifier] = enum_value.permissible_values![qualifier] ?? {};
      });
    });
    super(qualifier_objects);
    this._modify = underscore;
  }

  get objects(): BioLinkQualifiersObject {
    return this._objects_in_tree;
  }

  protected addNewObjectToTree(name: string): void {
    this._objects_in_tree[this._modify(name)] = new Qualifier(this._modify(name), this._objects_in_yaml[name]);
  }

  getDescendants(name: string): BioLinkQualifierObject[] {
    return super.getDescendants(name) as BioLinkQualifierObject[];
  }

  getAncestors(name: string): BioLinkQualifierObject[] {
    return super.getAncestors(name) as BioLinkQualifierObject[];
  }

  getPath(downstreamNode: string, upstreamNode: string): BioLinkQualifierObject[] {
    return super.getPath(downstreamNode, upstreamNode) as BioLinkQualifierObject[];
  }
}
