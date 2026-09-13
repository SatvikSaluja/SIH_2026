import { Router, type IRouter } from "express";
import healthRouter from "./health";
import bhoomiRouter from "./bhoomi";

const router: IRouter = Router();

router.use(healthRouter);
router.use(bhoomiRouter);

export default router;
