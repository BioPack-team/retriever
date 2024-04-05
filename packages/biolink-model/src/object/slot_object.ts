import { BioLinkSlot, BioLinkSlotObject } from '../types/types';
import { underscore } from '../utils';
import { pascalCase } from 'pascal-case';
import BaseObject from './base_object';

export default class Slot extends BaseObject implements BioLinkSlotObject {
  private _domain: string;
  private _range: string;
  private _inverse: string;
  private _exact_mapping: string[];
  private _close_mapping: string[];
  private _narrow_mapping: string[];
  private _broad_mapping: string[];
  private _symmetric: boolean;

  constructor(name: string, info: BioLinkSlot) {
    super(name, info);
    this._parent = typeof info.is_a === 'undefined' ? info.is_a : underscore(info.is_a);
    this._inverse = typeof info.inverse === 'undefined' ? info.inverse : underscore(info.inverse);
    this._domain = typeof info.domain === 'undefined' ? undefined : pascalCase(info.domain);
    this._range = typeof info.range === 'undefined' ? undefined : pascalCase(info.range);
    this._symmetric = typeof info.symmetric === 'undefined' ? false : true;
    this._exact_mapping = info.exact_mappings;
    this._close_mapping = info.close_mappings;
    this._narrow_mapping = info.narrow_mappings;
    this._broad_mapping = info.broad_mappings;
  }

  get inverse(): string {
    return this._inverse;
  }

  set inverse(value: string) {
    this._inverse = value;
  }

  get symmetric(): boolean {
    return this._symmetric;
  }

  get domain(): string {
    return this._domain;
  }

  get range(): string {
    return this._range;
  }

  get exact_mapping(): string[] {
    return this._exact_mapping;
  }

  get narrow_mapping(): string[] {
    return this._narrow_mapping;
  }

  get close_mapping(): string[] {
    return this._close_mapping;
  }

  get broad_mapping(): string[] {
    return this._broad_mapping;
  }

  addChild(child: string): void {
    this._children.push(underscore(child));
  }
}
