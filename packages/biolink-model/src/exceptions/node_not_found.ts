export default class NodeNotFoundError extends Error {
  constructor(message = 'Node is not in the tree') {
    super();
    this.name = 'NodeNotFound';
    this.message = message;
  }
}
