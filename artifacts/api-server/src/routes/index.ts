import { Router, type IRouter } from "express";
import healthRouter from "./health";
import sentinelRouter from "./sentinel";

// bhoomi.ts used to live here: the region/parcel/processing-run translation
// over the Python backend, plus an allowlisted passthrough for everything
// else. It moved into geocadastra/api/console.py, because production routes
// /api/* straight at that backend and never through this server -- so every
// route implemented only here 404'd once deployed while passing locally.
// The frontend's dev proxy now points at the Python backend directly, the
// same way the Netlify redirect does.
//
// What is left is Geo-VLM Sentinel, which has its own store and its own
// Gemini client and no equivalent on the Python side yet.
const router: IRouter = Router();

router.use(healthRouter);
router.use(sentinelRouter);

export default router;
