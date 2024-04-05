/* eslint-disable @typescript-eslint/no-var-requires */
import { promises as fs } from 'fs';
import path from 'path';
import async from 'async';
import { TrapiQueryGraph, CompactQualifiers } from '../types';

export interface TemplateLookup {
  subject: string;
  object: string;
  predicate: string;
  qualifiers: {
    [qualifierType: string]: string;
  };
}

export interface MatchedTemplate {
  template: string;
  queryGraph: TrapiQueryGraph;
}

export interface TemplateGroup {
  name: string;
  subject: string[];
  predicate: string[];
  object: string[];
  qualifiers?: CompactQualifiers;
  templates: string[];
}

export interface CompactEdge {
  subject: string;
  predicate: string;
  object: string;
  qualifiers: CompactQualifiers;
}

export async function getTemplates(lookups: TemplateLookup[]): Promise<MatchedTemplate[]> {
  async function getFiles(dir: string): Promise<string[]> {
    const rootFiles = await fs.readdir(path.resolve(dir));
    return await async.reduce(rootFiles, [] as string[], async (arr, fname: string) => {
      const fPath = path.join(dir, fname);
      if ((await fs.lstat(fPath)).isDirectory()) {
        return [...arr, ...(await getFiles(fPath))];
      }
      return [...arr, fPath];
    });
  }
  const templatePathsOnly = await getFiles(path.resolve(__dirname, '../../data/templates'));
  const templatePaths: { [fileBaseName: string]: string } = Object.fromEntries(
    templatePathsOnly.map((templatePath: string) => {
      return [path.basename(templatePath), templatePath];
    }),
  );
  const templateGroups = JSON.parse(
    await fs.readFile(path.resolve(__dirname, '../../data/templateGroups.json'), { encoding: 'utf8' }),
  );
  const matchingTemplatePaths: string[] = templateGroups.reduce((matches: string[], group: TemplateGroup) => {
    const lookupMatch = lookups.some((lookup) => {
      return (
        group.subject.includes(lookup.subject) &&
        group.object.includes(lookup.object) &&
        group.predicate.includes(lookup.predicate) &&
        Object.entries(lookup.qualifiers || {}).every(([qualifierType, qualifierValue]) => {
          return (group.qualifiers || {})[qualifierType] && group.qualifiers[qualifierType] === qualifierValue;
        })
      );
    });

    if (lookupMatch) {
      group.templates.forEach((template) => {
        if (!matches.includes(templatePaths[template])) {
          matches.push(templatePaths[template]);
        }
      });
    }
    return matches;
  }, [] as string[]);
  return await async.map(matchingTemplatePaths, async (templatePath: string) => {
    return {
      template: templatePath.substring(templatePath.lastIndexOf('/') + 1),
      queryGraph: JSON.parse(await fs.readFile(templatePath, { encoding: 'utf8' })).message.query_graph,
    };
  });
}

export async function supportedLookups(): Promise<CompactEdge[]> {
  const edges: Set<CompactEdge> = new Set();
  const templateGroups = JSON.parse(
    await fs.readFile(path.resolve(__dirname, '../../data/templateGroups.json'), { encoding: 'utf8' }),
  );
  templateGroups.forEach((group: TemplateGroup) => {
    group.subject.forEach((subject) => {
      group.predicate.forEach((predicate) => {
        group.object.forEach((object) => {
          edges.add({
            subject: `biolink:${subject}`,
            predicate: `biolink:${predicate}`,
            object: `biolink:${object}`,
            qualifiers: group.qualifiers,
          });
        });
      });
    });
  });
  return [...edges];
}
