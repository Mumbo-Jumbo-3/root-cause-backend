import { Container, getContainer } from "@cloudflare/containers";

interface Env {
  BACKEND: DurableObjectNamespace<Backend>;
  [binding: string]: unknown;
}

/** Runs the FastAPI server (see Dockerfile) as a single long-lived instance. */
export class Backend extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "30m";
  pingEndpoint = "localhost/ok";

  constructor(ctx: DurableObjectState<{}>, env: Env) {
    super(ctx, env);
    // Every Worker var and secret becomes an environment variable in the container,
    // so `wrangler secret put` is the only place backend config lives.
    this.envVars = Object.fromEntries(
      Object.entries(env).filter((entry): entry is [string, string] => typeof entry[1] === "string"),
    );
  }
}

export default {
  fetch(request, env) {
    return getContainer(env.BACKEND).fetch(request);
  },
} satisfies ExportedHandler<Env>;
