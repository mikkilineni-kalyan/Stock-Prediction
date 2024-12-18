declare module 'react-plotly.js' {
    import { Component } from 'react';
    import { Data, Layout, Config } from 'plotly.js';

    interface PlotParams {
        data: Data[];
        layout?: Partial<Layout>;
        config?: Partial<Config>;
        frames?: object[];
        style?: object;
        useResizeHandler?: boolean;
        debug?: boolean;
        onInitialized?: (figure: object) => void;
        onUpdate?: (figure: object) => void;
        onPurge?: (figure: object) => void;
        onError?: (err: Error) => void;
    }

    export default class Plot extends Component<PlotParams> {}
} 