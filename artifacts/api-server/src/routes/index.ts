import { Router, type IRouter } from "express";
import healthRouter from "./health";
import bhoomiRouter from "./bhoomi";
import sentinelRouter from "./sentinel";

const router: IRouter = Router();

router.use(healthRouter);
router.use(bhoomiRouter);
router.use(sentinelRouter);

export default router;
