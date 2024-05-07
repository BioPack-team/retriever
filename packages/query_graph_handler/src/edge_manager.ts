import _ from 'lodash';
import { LogEntry, StampedLog } from '@retriever/utils';
import { RetrieverError } from '@retriever/types';
import Debug from 'debug';
const debug = Debug('retriever:edge-manager');
import * as config from './config';
import BatchEdgeQueryHandler, { BatchEdgeQueryOptions } from './batch_edge_query';
import { Telemetry } from '@retriever/utils';
import { Record, QEdge } from '@retriever/graph';
import MetaKG from '@retriever/smartapi-kg';
import { QueryHandlerOptions } from '@retriever/types';
import { UnavailableAPITracker } from './types';
import { RecordsByQEdgeID } from './results_assembly/query_results';
import path from 'path';
import { promises as fs } from 'fs';

const MAX_RECORDS_PER_EDGE = parseInt(process.env.MAX_RECORDS_PER_EDGE ?? '100000');
const MAX_RECORDS_PER_QUERY = parseInt(process.env.MAX_RECORDS_PER_EDGE ?? '200000');

export default class QueryEdgeManager {
  private _qEdges: QEdge[];
  private _metaKG: MetaKG;
  logs: StampedLog[];
  private _records: Record[];
  options: QueryHandlerOptions;
  private _organizedRecords: RecordsByQEdgeID;
  constructor(edges: QEdge[], metaKG: MetaKG, options: QueryHandlerOptions) {
    // flatten list of all edges available
    this._qEdges = _.flatten(edges);
    this._metaKG = metaKG;
    this.logs = [];
    this._records = [];
    //organized by edge with refs to connected edges
    this._organizedRecords = {};
    this.options = options;
    this.init();
  }

  getRecords(): Record[] {
    debug(`(13) Edge Manager reporting combined records...`);
    return this._records;
  }

  getOrganizedRecords(): RecordsByQEdgeID {
    debug(`(13) Edge Manager reporting organized records...`);
    return this._organizedRecords;
  }

  init(): void {
    debug(`(3) Edge manager is managing ${this._qEdges.length} qEdges.`);
    this.logs.push(new LogEntry('DEBUG', null, `Edge manager is managing ${this._qEdges.length} qEdges.`).getLog());
  }

  getNext(): QEdge {
    //returns next edge with lowest entity count on
    //either object or subject OR no count last
    // available not yet executed
    const available_edges = this._qEdges.filter((qEdge) => !qEdge.executed);
    //safeguard for making sure there's available
    //edges when calling getNext
    if (available_edges.length == 0) {
      debug(`(5) Error: ${available_edges} available qEdges found.`);
      this.logs.push(
        new LogEntry(
          'DEBUG',
          null,
          `Edge manager cannot get next qEdge, ` + `(${available_edges}) available edges found.`,
        ).getLog(),
      );
    }
    //begin search
    let nextQEdge: QEdge;
    let lowest_entity_count: number;
    let current_obj_lowest = 0;
    let current_sub_lowest = 0;
    available_edges.forEach((qEdge) => {
      if (qEdge && qEdge.object.entity_count) {
        current_obj_lowest = qEdge.object.entity_count;
        if (!lowest_entity_count) {
          //set current lowest if none
          lowest_entity_count = current_obj_lowest;
        }
        if (current_obj_lowest <= lowest_entity_count) {
          //lowest is now object count
          nextQEdge = qEdge;
        }
      }
      if (qEdge && qEdge.subject.entity_count && qEdge.subject.entity_count > 0) {
        current_sub_lowest = qEdge.subject.entity_count;
        if (!lowest_entity_count) {
          //set current lowest if none
          lowest_entity_count = current_sub_lowest;
        }
        if (current_sub_lowest <= lowest_entity_count) {
          //lowest is now subject count
          nextQEdge = qEdge;
        }
      }
    });
    if (!nextQEdge) {
      //if no edge with count found pick the first empty
      //edge available
      const all_empty = available_edges.filter((edge) => !edge.object.entity_count && !edge.subject.entity_count);
      if (all_empty.length == 0) {
        debug(`(5) Error: No available qEdges found.`);
        this.logs.push(new LogEntry('DEBUG', null, `Cannot get next edge, No available qEdges found.`).getLog());
      }
      debug(`(5) Sending next edge '${all_empty[0].getID()}' with NO entity count.`);
      return this.preSendOffCheck(all_empty[0]);
    }
    debug(
      `(5) Sending next edge '${nextQEdge.getID()}' ` +
        `WITH entity count...(${nextQEdge.subject.entity_count || nextQEdge.object.entity_count})`,
    );
    return this.preSendOffCheck(nextQEdge);
  }

  logEntityCounts(): void {
    this._qEdges.forEach((qEdge) => {
      debug(
        `'${qEdge.getID()}'` +
          ` : (${qEdge.subject.entity_count || 0}) ` +
          `${qEdge.reverse ? '<--' : '-->'}` +
          ` (${qEdge.object.entity_count || 0})`,
      );
    });
  }

  _logSkippedQueries(unavailableAPIs: UnavailableAPITracker): void {
    Object.entries(unavailableAPIs).forEach(([api, { skippedQueries }]) => {
      if (skippedQueries > 0) {
        const skipMessage = `${skippedQueries} additional quer${skippedQueries > 1 ? 'ies' : 'y'} to ${api} ${
          skippedQueries > 1 ? 'were' : 'was'
        } skipped as the API was unavailable.`;
        debug(skipMessage);
        this.logs.push(new LogEntry('WARNING', null, skipMessage).getLog());
      }
    });
  }

  checkEntityMax(nextQEdge: QEdge): void {
    const max = config.ENTITY_MAX;
    //(MAX) --- (0) not allowed
    //(MAX) --- (MAX) not allowed
    //(MAX) --- (2) allowed, (2 will be used)
    const sub_count = nextQEdge.object.getEntityCount();
    const obj_count = nextQEdge.subject.getEntityCount();
    debug(`Checking entity max : (${sub_count})--(${obj_count})`);
    if (
      (obj_count == 0 && sub_count > max) ||
      (obj_count > max && sub_count == 0) ||
      (obj_count > max && sub_count > max)
    ) {
      throw new RetrieverError(`Max number of entities exceeded (${max}) in '${nextQEdge.getID()}'`);
    }
  }

  preSendOffCheck(nextQEdge: QEdge): QEdge {
    // next: qEdge
    //check that edge entities are or have potential to stay
    //under max limit
    this.checkEntityMax(nextQEdge);
    if (nextQEdge.object.entity_count && nextQEdge.subject.entity_count) {
      //if at the time of being queried the edge has both
      //obj and sub entity counts
      //chose obj/suj lower entity count for query
      nextQEdge.chooseLowerEntityValue();
      this.logs.push(new LogEntry('DEBUG', null, `Next qEdge will pick lower entity value to use for query.`).getLog());
    } else if (
      (nextQEdge.object.entity_count && !nextQEdge.subject.entity_count) ||
      (!nextQEdge.object.entity_count && !nextQEdge.subject.entity_count)
    ) {
      debug(`(5) Checking direction of edge with one set of entities...`);
      //check direction is correct if edge only has one set of entities
      //before sending off
      nextQEdge.reverse = nextQEdge.subject.entity_count ? false : true;
    }
    this.logs.push(
      new LogEntry('DEBUG', null, `Edge manager is sending next qEdge '${nextQEdge.getID()}' for execution.`).getLog(),
    );
    this.logEntityCounts();
    return nextQEdge;
  }

  getEdgesNotExecuted(): number {
    //simply returns a number of edges not marked as executed
    const found = this._qEdges.filter((edge) => !edge.executed);
    const not_executed = found.length;
    if (not_executed) debug(`(4) Edges not yet executed = ${not_executed}`);
    return not_executed;
  }

  _filterEdgeRecords(qEdge: QEdge): Record[] {
    const keep: Record[] = [];
    const records = qEdge.records;
    const subjectCuries = qEdge.subject.curie;
    const objectCuries = qEdge.object.curie;
    debug(
      `'${qEdge.getID()}' Reversed[${qEdge.reverse}] (${JSON.stringify(subjectCuries.length || 0)})` +
        `--(${JSON.stringify(objectCuries.length || 0)}) entities / (${records.length}) records.`,
    );
    // debug(`IDS SUB ${JSON.stringify(sub_count)}`)
    // debug(`IDS OBJ ${JSON.stringify(obj_count)}`)
    const execSubjectCuries = qEdge.reverse ? objectCuries : subjectCuries;
    const execObjectCuries = qEdge.reverse ? subjectCuries : objectCuries;

    records.forEach((record) => {
      // check against original, primaryID, and equivalent ids
      const subjectIDs = [record.subject.original, record.subject.curie, ...record.subject.equivalentCuries];
      const objectIDs = [record.object.original, record.object.curie, ...record.object.equivalentCuries];

      // there must be at least a minimal intersection
      const subjectMatch =
        subjectIDs.some((curie) => execSubjectCuries.includes(curie)) || execSubjectCuries.length === 0;
      const objectMatch = objectIDs.some((curie) => execObjectCuries.includes(curie)) || execObjectCuries.length === 0;

      //if both ends match then keep record

      // Don't keep self-edges
      const selfEdge = [...subjectIDs].some((curie) => objectIDs.includes(curie));
      if (subjectMatch && objectMatch && !selfEdge) {
        keep.push(record);
      }
    });
    debug(`'${qEdge.getID()}' dropped (${records.length - keep.length}) records.`);
    this.logs.push(
      new LogEntry(
        'DEBUG',
        null,
        `'${qEdge.getID()}' kept (${keep.length}) / dropped (${records.length - keep.length}) records.`,
      ).getLog(),
    );
    return keep;
  }

  collectRecords(): boolean {
    //go through edges and collect records organized by edge
    let recordsByQEdgeID: RecordsByQEdgeID = {};
    //all res merged
    let combinedRecords = [];
    let brokenChain = false;
    const brokenEdges = [];
    debug(`(11) Collecting records...`);
    //First: go through edges and filter that each edge is holding
    this._qEdges.forEach((qEdge) => {
      const qEdgeID = qEdge.getID();
      const filteredRecords = qEdge.records.map((record) => record.queryDirection());
      if (filteredRecords.length == 0) {
        this.logs.push(new LogEntry('WARNING', null, `Warning: qEdge '${qEdgeID}' resulted in (0) records.`).getLog());
        brokenChain = true;
        brokenEdges.push(qEdgeID);
      }
      this.logs = [...this.logs, ...qEdge.logs];
      //collect records
      combinedRecords = combinedRecords.concat(filteredRecords);
      let connections = qEdge.subject.getConnections().concat(qEdge.object.getConnections());
      connections = connections.filter((id) => id !== qEdgeID);
      connections = [...new Set(connections)];
      recordsByQEdgeID[qEdgeID] = {
        records: filteredRecords,
        connected_to: connections,
      };
      debug(`(11) '${qEdgeID}' keeps (${filteredRecords.length}) records!`);
      this.logs.push(new LogEntry('INFO', null, `'${qEdgeID}' keeps (${filteredRecords.length}) records!`).getLog());
      debug(`----------`);
    });
    if (brokenChain) {
      recordsByQEdgeID = {};
      this.logs.push(
        new LogEntry(
          'WARNING',
          null,
          `qEdges ${brokenEdges} resulted in (0) records. No complete paths can be formed.`,
        ).getLog(),
      );
      debug(`(12) qEdges ${brokenEdges} resulted in (0) records. No complete paths can be formed.`);
    }
    //Organized by edge: update query records
    this._organizedRecords = recordsByQEdgeID;
    debug(`(12) Collected records for: ${JSON.stringify(Object.keys(this._organizedRecords))}!`);
    //Combined: update query_graph
    this._records = combinedRecords;
    // var fs = require('fs');
    // fs.writeFile("organized_records.json", JSON.stringify(records, function( key, value) {
    //     if( key == 'records') { return '$records'}
    //     else {return value;}
    //     }), function(err) {
    //     if (err) {
    //         console.log(err);
    //     }
    // });
    if (!brokenChain) {
      debug(`(12) Collected (${this._records.length}) records!`);
      this.logs.push(new LogEntry('DEBUG', null, `Edge manager collected (${this._records.length}) records!`).getLog());
    }

    return !brokenChain;
  }

  updateEdgeRecords(currentQEdge: QEdge): void {
    //1. filter edge records based on current status
    const filteredRecords = this._filterEdgeRecords(currentQEdge);
    //2.trigger node update / entity update based on new status
    currentQEdge.storeRecords(filteredRecords);
  }

  updateAllOtherEdges(currentQEdge: QEdge): void {
    //update and filter all other edges
    debug(`Updating all other edges...`);
    const currentQEdgeID = currentQEdge.getID();
    this._qEdges.forEach((qEdge) => {
      if (qEdge.getID() !== currentQEdgeID && qEdge.records.length) {
        debug(`Updating "${qEdge.getID()}"...`);
        this.updateEdgeRecords(qEdge);
        this.updateEdgeRecords(currentQEdge);
      }
    });
  }

  _createBatchQueryHandler(qEdge: QEdge, metaKG: MetaKG): BatchEdgeQueryHandler {
    const handler = new BatchEdgeQueryHandler(metaKG, {
      ...this.options,
      recordHashEdgeAttributes: config.EDGE_ATTRIBUTES_USED_IN_RECORD_HASH,
    } as BatchEdgeQueryOptions);
    handler.setEdges(qEdge);
    return handler;
  }

  async dumpRecords(records: Record[]): Promise<void> {
    let filePath = path.resolve('../../..', process.env.DUMP_RECORDS);
    // create new (unique) file if arg is directory
    try {
      if ((await fs.lstat(filePath)).isDirectory()) {
        filePath = path.resolve(filePath, `recordDump-${new Date().toISOString()}.json`);
      }
    } catch (e) {
      null; // specified a file, which doesn't exist (which is fine)
    }
    let direction = false;
    if (process.env.DUMP_RECORDS_DIRECTION?.includes('exec')) {
      direction = true;
      records = [...records].map((record) => record.queryDirection());
    }
    await fs.writeFile(filePath, JSON.stringify(records.map((record) => record.freeze())));
    const logMessage = `Dumping Records ${direction ? `(in execution direction)` : ''} to ${filePath}`;
    debug(logMessage);
  }

  async executeEdges(): Promise<boolean> {
    const unavailableAPIs: UnavailableAPITracker = {};
    while (this.getEdgesNotExecuted()) {
      const span = Telemetry.startSpan({ description: 'edgeExecution' });
      //next available/most efficient edge
      const currentQEdge = this.getNext();
      //crate queries from edge
      const queryBatchHandler = this._createBatchQueryHandler(currentQEdge, this._metaKG);
      this.logs.push(
        new LogEntry(
          'INFO',
          null,
          `Executing ${currentQEdge.getID()}${currentQEdge.isReversed() ? ' (reversed)' : ''}: ${
            currentQEdge.subject.id
          } ${currentQEdge.isReversed() ? '<--' : '-->'} ${currentQEdge.object.id}`,
        ).getLog(),
      );
      debug(`(5) Executing current edge >> "${currentQEdge.getID()}"`);
      //execute current edge query
      let queryRecords = await queryBatchHandler.query(queryBatchHandler.qEdges, unavailableAPIs);
      this.logs = [...this.logs, ...queryBatchHandler.logs];
      if (queryRecords === undefined) return;
      // create an edge execution summary
      let success = 0,
        fail = 0,
        total = 0;
      const cached = this.logs.filter(
        ({ data }) => data?.qEdgeID === currentQEdge.id && data?.type === 'cacheHit',
      ).length;
      total += cached;
      this.logs
        .filter(({ data }) => data?.qEdgeID === currentQEdge.id && data?.type === 'query')
        .forEach(({ data }) => {
          !data.error ? success++ : fail++;
          total++;
        });
      this.logs.push(
        new LogEntry(
          'INFO',
          null,
          `${currentQEdge.id} execution: ${total} queries (${success} success/${fail} fail/${cached} cached) return (${queryRecords.length}) records`,
          {},
        ).getLog(),
      );
      if (queryRecords.length === 0) {
        this._logSkippedQueries(unavailableAPIs);
        debug(`(X) Terminating..."${currentQEdge.getID()}" got 0 records.`);
        this.logs.push(
          new LogEntry(
            'WARNING',
            null,
            `qEdge (${currentQEdge.getID()}) got 0 records. Your query terminates.`,
          ).getLog(),
        );
        span.finish();
        return;
      }

      // Check if record count threatens stability
      const totalRecords =
        this._qEdges.reduce((total, qEdge) => {
          return total + qEdge.records.length;
        }, 0) + queryRecords.length;
      let maxRecordsMessage = [
        `Qedge ${currentQEdge.id}`,
        `obtained ${queryRecords.length} records,`,
        queryRecords.length === MAX_RECORDS_PER_EDGE ? 'meeting' : 'exceeding',
        `maximum of ${MAX_RECORDS_PER_QUERY}`,
        `Truncating records for this edge to ${MAX_RECORDS_PER_EDGE}.`,
        `Your query may be too general?`,
      ];
      if (totalRecords > MAX_RECORDS_PER_QUERY) {
        maxRecordsMessage = maxRecordsMessage.slice(0, 2);
        maxRecordsMessage.push(
          ...[
            `totalling ${totalRecords} for this query.`,
            `This exceeds the per-query maximum of ${MAX_RECORDS_PER_QUERY}.`,
            `For stability purposes, this query is terminated.`,
            `Please consider further refining your query.`,
          ],
        );
        debug(maxRecordsMessage.join(' '));
        this.logs.push(new LogEntry('WARNING', null, maxRecordsMessage.join(' ')).getLog());
        Telemetry.captureException(new Error(`Stopped on globalMaxRecords (exceeded ${MAX_RECORDS_PER_QUERY})`));
        return;
      }

      if (queryRecords.length > MAX_RECORDS_PER_EDGE) {
        debug(maxRecordsMessage.join(' '));
        this.logs.push(new LogEntry('WARNING', null, maxRecordsMessage.join(' ')).getLog());

        queryRecords = queryRecords.slice(0, MAX_RECORDS_PER_EDGE);
      }

      // storing records will trigger a node entity count update
      currentQEdge.storeRecords(queryRecords);

      const span1 = Telemetry.startSpan({ description: 'filteringRecords' });
      // filter records
      this.updateEdgeRecords(currentQEdge);
      span1?.finish();

      const span2 = Telemetry.startSpan({ description: 'updatingRecordEdges' });

      // update and filter neighbors
      this.updateAllOtherEdges(currentQEdge);
      span2?.finish();

      // check that any records are kept
      if (!currentQEdge.records.length) {
        this._logSkippedQueries(unavailableAPIs);
        debug(`(X) Terminating..."${currentQEdge.getID()}" kept 0 records.`);
        this.logs.push(
          new LogEntry(
            'WARNING',
            null,
            `qEdge (${currentQEdge.getID()}) kept 0 records. Your query terminates.`,
          ).getLog(),
        );
        span.finish();
        return;
      }
      // edge all done
      currentQEdge.executed = true;
      debug(`(10) Edge successfully queried.`);
      span.finish();
    }
    this._logSkippedQueries(unavailableAPIs);
    // collect and organize records
    if (!this.collectRecords()) {
      debug(`(X) Terminating...No complete paths.`);
      this.logs.push(
        new LogEntry('WARNING', null, `No complete paths could be formed. Your query terminates.`).getLog(),
      );
      return;
    }
    // dump records if set to do so
    if (process.env.DUMP_RECORDS) {
      await this.dumpRecords(this.getRecords());
    }
    return true;
  }
}
