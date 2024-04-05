import Slot from '../object/slot_object';
import { underscore } from '../utils';
import { BioLinkSlots, BioLinkSlotObject, BioLinkSlotsObject, BioLinkSlotTreeObject } from '../types/types';
import BaseTree from './base_tree';

export default class BioLinkClassTree extends BaseTree implements BioLinkSlotTreeObject {
  protected _objects_in_yaml: BioLinkSlots;
  protected _objects_in_tree: BioLinkSlotsObject;

  constructor(objects: BioLinkSlots) {
    super(objects);
    this._modify = underscore;
  }

  get objects(): BioLinkSlotsObject {
    return this._objects_in_tree;
  }

  protected addNewObjectToTree(name: string): void {
    this._objects_in_tree[this._modify(name)] = new Slot(this._modify(name), this._objects_in_yaml[name]);
  }

  construct() {
    super.construct();
    Object.keys(this._objects_in_tree).map((name: string) => this.inferInverseRelationship(name));
  }

  getDescendants(name: string): BioLinkSlotObject[] {
    return super.getDescendants(name) as BioLinkSlotObject[];
  }

  getAncestors(name: string): BioLinkSlotObject[] {
    return super.getAncestors(name) as BioLinkSlotObject[];
  }

  getPath(downstreamNode: string, upstreamNode: string): BioLinkSlotObject[] {
    return super.getPath(downstreamNode, upstreamNode) as BioLinkSlotObject[];
  }

  protected inferInverseRelationship(name: string) {
    if (typeof this._objects_in_tree[this._modify(name)].inverse === 'undefined') {
      const inverse = Object.values(this._objects_in_tree).find(
        (obj: BioLinkSlotObject) => obj.inverse === this._modify(name),
      );
      if (typeof inverse !== 'undefined') {
        this._objects_in_tree[this._modify(name)].inverse = inverse.name;
      }
    }
  }
}
