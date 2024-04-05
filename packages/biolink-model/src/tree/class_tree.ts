import Entity from '../object/entity_object';
import { pascalCase } from 'pascal-case';
import {
  BioLinkClassesObject,
  BioLinkEntityObject,
  BioLinkEntitiesObject,
  BioLinkClassTreeObject,
} from '../types/types';
import BaseTree from './base_tree';

export default class BioLinkClassTree extends BaseTree implements BioLinkClassTreeObject {
  protected _objects_in_yaml: BioLinkClassesObject;
  protected _objects_in_tree: BioLinkEntitiesObject;

  constructor(objects: BioLinkClassesObject) {
    super(objects);
    this._modify = pascalCase;
  }

  get objects(): BioLinkEntitiesObject {
    return this._objects_in_tree;
  }

  protected addNewObjectToTree(name: string): void {
    this._objects_in_tree[this._modify(name)] = new Entity(this._modify(name), this._objects_in_yaml[name]);
  }

  getDescendants(name: string): BioLinkEntityObject[] {
    return super.getDescendants(name) as BioLinkEntityObject[];
  }

  getAncestors(name: string): BioLinkEntityObject[] {
    return super.getAncestors(name) as BioLinkEntityObject[];
  }

  getPath(downstreamNode: string, upstreamNode: string): BioLinkEntityObject[] {
    return super.getPath(downstreamNode, upstreamNode) as BioLinkEntityObject[];
  }
}
