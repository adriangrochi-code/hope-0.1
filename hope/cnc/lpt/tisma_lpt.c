/* =================================================================================================
 * tisma_lpt.c — TISMA CNC step/direction pulse engine for the PC parallel port (LPT).
 * =================================================================================================
 *
 * WHAT IT IS
 *   The piece that replaces Mach3's parallel port driver: it turns the motion TISMA has already
 *   planned into step and direction pulses on the DB25 pins, reads the inputs (limit and home
 *   switches, probe, emergency stop) and drives the outputs (enable, spindle relay and PWM,
 *   coolant, charge pump). It runs as a separate small program so that its timing loop is not
 *   slowed down by Python (garbage collector, GIL, the web server).
 *
 *   Builds (see compilar.sh):
 *     tisma_lpt32.exe   Windows XP .. 11, 32-bit (also runs on 64-bit Windows through WOW64).
 *     tisma_lpt64.exe   Windows XP x64 .. 11, 64-bit.
 *     tisma_lpt_sim     Linux/any: the same engine with a SIMULATED port and VIRTUAL time, for the
 *                       tests (every pulse is counted, nothing waits for the clock).
 *
 * HOW IT FITS
 *   hope/cnc/lpt.py (LptLink) starts it, sends the configuration and the planned blocks as text
 *   lines on stdin and reads its reports on stdout. The interpreter and the planner (acceleration,
 *   junction speeds) stay in Python; here only the real-time part lives: one «kernel tick» every
 *   1/FREQ seconds (25 kHz by default, like Mach3's kernel speed), in which each axis may give at
 *   most one step.
 *
 *   Port access on Windows goes through InpOut32 / InpOutx64 (free driver by Highrez, the one many
 *   Mach3-era tools use): inpout32.dll (32-bit build) or inpoutx64.dll (64-bit build) next to the
 *   exe. Its kernel driver is installed the first time it runs with administrator rights.
 *
 * TIMING (why it is done this way)
 *   Windows is not a real-time system: a thread can be interrupted for several milliseconds.
 *   The tick loop therefore busy-waits on QueryPerformanceCounter in a TIME_CRITICAL thread pinned
 *   to the last CPU core, with the process at REALTIME (or HIGH without admin) priority. Motion
 *   time is counted in ticks, not wall-clock time: if the loop is late by more than one tick it
 *   does NOT catch up with a burst of pulses (that would demand an impossible acceleration from the
 *   motors); it resynchronises and the machine simply moves a little later. Late ticks are counted
 *   («overruns» in the status) and the latency test (TEST) tells TISMA how fast this PC can go.
 *   On single-core PCs the busy wait would freeze everything else: TISMA refuses them unless the
 *   profile allows it explicitly (see lpt.py).
 *
 * MOTION (grbl-style live profile)
 *   Each block arrives with its length L (mm, or degrees on a rotary-only move), cruise speed vc,
 *   exit speed ve (the junction speed the planner allowed into the next block), acceleration a and
 *   the steps per axis. Every tick:
 *       v = min(v + a·dt, vc·override, sqrt(ve² + 2·a·(L − s)))     (accelerate, cruise or brake)
 *       s = s + v·dt
 *   and axis i must have given floor(|steps_i| · s / L) steps (a DDA: every axis lands exactly on
 *   its step count at the end of the block, all of them in a straight line). Feed hold brakes with
 *   the same acceleration and stays inside the block; resume accelerates again from there.
 *
 * PROTOCOL (stdin, one command per line; numbers in decimal; ids are TISMA's)
 *   CFG FREQ hz | PULSE us | PORT addr(hex ok) | REPORT ms | DLL path | DRYRUN 0/1 | TRACE 0/1
 *   CFG AXIS i STEP pin DIR pin STEPLOW 0/1 DIRINV 0/1      i = 0..5 (X Y Z A B C)
 *   CFG OUT name pin activelow   names: ENABLE SPINDLE_CW SPINDLE_CCW FLOOD MIST PWM CHARGEPUMP
 *   CFG IN name pin activelow    names: ESTOP PROBE LIMIT0..LIMIT5 HOME0..HOME5
 *   CFG PWMFREQ hz | CPFREQ hz
 *   START                        configuration done: start the tick loop (prints READY)
 *   M id L vc ve a rapid s0 s1 s2 s3 s4 s5 [flags]    queued move; flags: P2/P3 (probe, alarm /
 *                                no alarm without contact), H<axis> (stop on that home switch),
 *                                I<mask> (ignore those limit inputs: backing off a switch)
 *   O id name value              queued output change (PWM value = duty in ‰)
 *   W id seconds                 queued dwell
 *   P id                         queued pause (M0/M1/M6): holds until RESUME
 *   SET name value               change an output NOW (coolant button with a program running)
 *   HOLD | RESUME | CANCEL       feed hold / continue / brake and discard the queue (no alarm)
 *   OVF pct | OVR pct            feed / rapid override (10–200 / 1–100)
 *   SETPOS s0 .. s5              set the step counters (after homing)
 *   RESET                        clear an alarm (only if the emergency stop is released)
 *   TEST seconds                 latency test (no motion): prints LAT
 *   QUIT
 *   Simulator only: «@t command» runs the command at virtual time t (s); «SIMIN name axis > steps»
 *   (or <) makes an input active when that axis passes the position; «IN name 0/1» forces an
 *   input; CFG TMAX s stops the run. «CFG INTERACTIVE speed» (Linux/macOS) makes the simulator
 *   a live controller instead: commands are read while it runs and virtual time goes «speed»
 *   times faster than the clock — what the Python tests of LptLink talk to.
 *
 * REPORTS (stdout)
 *   READY version bits                      after START
 *   S state done p0..p5 inputs v overruns   every REPORT ms: state I/R/H/A, last finished id,
 *                                           position (steps), active inputs (bit mask in the order
 *                                           of IN names), speed (units/s), late ticks so far
 *   PRB ok p0..p5                           probe result (position latched at the trigger)
 *   HOMED id                                a home switch stopped move «id»
 *   ALARM text                              motion stopped (limit, e-stop, probe without contact…)
 *   LAT ticks max_us p99_us p999_us         latency test result
 *   PULSE tick mask dirmask                 (simulator with TRACE 1) every step pulse
 *   PORT tick reg value                     (simulator with TRACE 2) every port register write
 *   ERR text                                bad command (ignored)
 *
 * UNITS: steps, seconds, «units» = mm (or ° for a rotary-only block), units/s, units/s².
 *
 * EXAMPLE (what lpt.py sends)
 *   CFG FREQ 25000
 *   CFG AXIS 0 STEP 2 DIR 3 STEPLOW 0 DIRINV 0
 *   START
 *   M 1 10 20 0 200 0 8000 0 0 0 0 0         10 mm in X at 20 mm/s, 800 steps/mm
 * ================================================================================================= */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TISMA_SIM
#  define REAL_PORT 0
#else
#  ifndef _WIN32
#    error "the real build is for Windows (mingw); build with -DTISMA_SIM for the simulator"
#  endif
#  define REAL_PORT 1
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0501            /* Windows XP: no API newer than XP is used */
#  endif
#  include <windows.h>
#  include <mmsystem.h>
#endif

#define VERSION   "1.0"
#define NAXES     6
#define QSIZE     512                       /* queued items (moves, outputs, dwells, pauses) */
#define CMDQ      256                       /* commands waiting for the tick thread */

/* ==================================================================================================
 * ========================================= DB25 PIN MAP ===========================================
 * ==================================================================================================
 * Register 0 = data (base), 1 = status (base+1, inputs), 2 = control (base+2).
 * «hwinv»: the port hardware inverts that pin (status bit 7 = pin 11; control bits 0, 1, 3 =
 * pins 1, 14, 17), so the register bit is the opposite of the electrical level. */
typedef struct { int reg, bit, hwinv, output; } PinDef;
static const PinDef PINS[26] = {
    {-1, 0, 0, 0},                                              /* 0: no pin */
    {2, 0, 1, 1},                                               /* 1: strobe */
    {0, 0, 0, 1}, {0, 1, 0, 1}, {0, 2, 0, 1}, {0, 3, 0, 1},     /* 2..5: data 0..3 */
    {0, 4, 0, 1}, {0, 5, 0, 1}, {0, 6, 0, 1}, {0, 7, 0, 1},     /* 6..9: data 4..7 */
    {1, 6, 0, 0},                                               /* 10: ack */
    {1, 7, 1, 0},                                               /* 11: busy (inverted) */
    {1, 5, 0, 0},                                               /* 12: paper end */
    {1, 4, 0, 0},                                               /* 13: select */
    {2, 1, 1, 1},                                               /* 14: auto feed */
    {1, 3, 0, 0},                                               /* 15: error */
    {2, 2, 0, 1},                                               /* 16: init */
    {2, 3, 1, 1},                                               /* 17: select in */
    {-1, 0, 0, 0}, {-1, 0, 0, 0}, {-1, 0, 0, 0}, {-1, 0, 0, 0}, /* 18..25: ground */
    {-1, 0, 0, 0}, {-1, 0, 0, 0}, {-1, 0, 0, 0}, {-1, 0, 0, 0},
};

/* One configured pin: number (0 = not used) and whether «active» is the LOW electrical level
 * (most breakout boards drive optocouplers: active low is common for enable and inputs). */
typedef struct { int pin, activelow; } Pin;

/* --- outputs and inputs by name --------------------------------------------------------------- */
enum { OUT_ENABLE, OUT_SPINDLE_CW, OUT_SPINDLE_CCW, OUT_FLOOD, OUT_MIST, OUT_PWM, OUT_CHARGEPUMP, NOUT };
static const char *OUT_NAMES[NOUT] = {"ENABLE", "SPINDLE_CW", "SPINDLE_CCW", "FLOOD", "MIST", "PWM", "CHARGEPUMP"};
enum { IN_ESTOP, IN_PROBE, IN_LIMIT0, IN_HOME0 = IN_LIMIT0 + NAXES, NIN = IN_HOME0 + NAXES };

static int in_index(const char *name) {
    /* Input name → index (ESTOP, PROBE, LIMIT0..5, HOME0..5); -1 if unknown. */
    if (!strcmp(name, "ESTOP")) return IN_ESTOP;
    if (!strcmp(name, "PROBE")) return IN_PROBE;
    if (!strncmp(name, "LIMIT", 5) && name[5] >= '0' && name[5] < '0' + NAXES && !name[6]) return IN_LIMIT0 + name[5] - '0';
    if (!strncmp(name, "HOME", 4) && name[4] >= '0' && name[4] < '0' + NAXES && !name[5]) return IN_HOME0 + name[4] - '0';
    return -1;
}

static int out_index(const char *name) {
    /* Output name → index; -1 if unknown. */
    int i;
    for (i = 0; i < NOUT; i++)
        if (!strcmp(name, OUT_NAMES[i])) return i;
    return -1;
}

/* ==================================================================================================
 * ========================================= CONFIGURATION ==========================================
 * ================================================================================================== */
typedef struct {
    /* Kernel frequency (Hz): ticks per second. Each axis can step at most once per tick, so the
     * maximum step rate is FREQ (one pulse per tick) — TISMA keeps axes at FREQ/2 for margin.
     * 25000 = Mach3's default «kernel speed». Higher needs a PC with low latency (TEST). */
    double freq;
    /* Step pulse width (µs): how long the step pin stays active. Drivers need 1–5 µs (DM542: 2.5;
     * old optocoupled boards 5–10). Must be well below 1/FREQ (40 µs at 25 kHz). Default 5. */
    double pulse_us;
    unsigned port;                          /* LPT base address: 0x378 (LPT1), 0x278, PCI cards e.g. 0xD010 */
    double report_ms;                       /* status report period (ms); 20 = 50 per second */
    double pwm_freq;                        /* spindle PWM frequency (Hz); resolution = FREQ / pwm_freq levels */
    double cp_freq;                         /* charge pump frequency (Hz); boards expect 12.5 kHz (≤ FREQ/2) */
    int dryrun;                             /* 1 = do not touch the port (try the exe without hardware) */
    int trace;                              /* simulator: print every pulse */
    double tmax;                            /* simulator: stop after this virtual time (s) */
    double interactive;                     /* simulator: >0 = live mode, virtual time × this */
    char dll[260];
    Pin step[NAXES], dir[NAXES];
    int dirinv[NAXES];
    Pin out[NOUT], in[NIN];
} Config;

/* ==================================================================================================
 * ============================================ QUEUE ===============================================
 * ================================================================================================== */
enum { IT_MOVE, IT_OUT, IT_DWELL, IT_PAUSE };
typedef struct {
    int type;
    long id;
    double len, vc, ve, acc;                /* move: units, units/s, units/s, units/s² */
    int rapid;
    long steps[NAXES];                      /* move: signed steps per axis */
    int probe;                              /* 0, 2 (G38.2) or 3 (G38.3) */
    int home_axis;                          /* -1 or axis whose HOME input ends the move */
    int ignore_limits;                      /* bit mask of LIMITi inputs ignored (backing off) */
    int out, value;                         /* output change */
    double seconds;                         /* dwell */
} Item;

/* ==================================================================================================
 * ============================================ ENGINE ==============================================
 * ================================================================================================== */
enum { ST_IDLE = 'I', ST_RUN = 'R', ST_HOLD = 'H', ST_ALARM = 'A' };

typedef struct {
    Config c;
    Item q[QSIZE];
    int qhead, qtail;                       /* items [qhead, qtail) are queued (owned by the tick thread) */
    Item cur;                               /* item running */
    int has_cur;
    double s, v;                            /* move: units done and current speed */
    long done_steps[NAXES];                 /* steps given inside the current move */
    long pos[NAXES];                        /* absolute step counters */
    int dirbits;                            /* current direction per axis (1 = negative) */
    int dir_wait;                           /* ticks to wait after a direction change (setup time) */
    int holding, cancelling, paused;
    int probe_hit;                          /* the probe triggered in this move: braking */
    double dwell_left;
    char state;
    long last_done;
    int outv[NOUT];                         /* logical output values (PWM: duty ‰) */
    int inputs;                             /* active inputs (bit i = input i) */
    double ovf, ovr;                        /* overrides, fraction */
    unsigned long long tick;
    unsigned long overruns;
    int pwm_count, cp_count, cp_level;
    unsigned char data, ctrl;               /* last values written */
    int written;
} Engine;

static Engine E;

/* --- output of reports ---------------------------------------------------------------------------- */
#if REAL_PORT
static CRITICAL_SECTION out_lock;           /* the tick thread and the reporter both print */
#  define OUT_LOCK()   EnterCriticalSection(&out_lock)
#  define OUT_UNLOCK() LeaveCriticalSection(&out_lock)
#else
#  define OUT_LOCK()
#  define OUT_UNLOCK()
#endif

/* Events (PRB, HOMED, ALARM) from the tick thread go through a small ring so the tick loop never
 * blocks on stdout (a slow pipe would otherwise stall the pulses). */
#define EVQ 64
static char evq[EVQ][160];
static volatile int ev_head, ev_tail;

static void event(const char *text) {
    /* Queue one report line (dropped if the ring is full: the status line repeats the state). */
    int next = (ev_tail + 1) % EVQ;
    if (next == ev_head) return;
    snprintf(evq[ev_tail], sizeof evq[0], "%s", text);
#if REAL_PORT
    MemoryBarrier();
#endif
    ev_tail = next;
}

static void flush_events(void) {
    /* Print queued events (reporter side). */
    while (ev_head != ev_tail) {
        OUT_LOCK();
        printf("%s\n", evq[ev_head]);
        OUT_UNLOCK();
        ev_head = (ev_head + 1) % EVQ;
    }
}

static void raise_alarm(const char *text) {
    /* Stop everything now (no braking: an alarm means something is wrong), switch the spindle and
     * coolant off, forget the queue and report. Positions stay as counted. */
    char buf[160];
    E.has_cur = 0;
    E.qhead = E.qtail;
    E.v = 0;
    E.holding = E.cancelling = E.paused = 0;
    E.outv[OUT_SPINDLE_CW] = E.outv[OUT_SPINDLE_CCW] = E.outv[OUT_FLOOD] = E.outv[OUT_MIST] = E.outv[OUT_PWM] = 0;
    E.state = ST_ALARM;
    snprintf(buf, sizeof buf, "ALARM %s", text);
    event(buf);
}

/* ==================================================================================================
 * =========================================== PORT I/O =============================================
 * ================================================================================================== */
#if REAL_PORT
typedef void (__stdcall *Out32Fn)(short, short);
typedef short (__stdcall *Inp32Fn)(short);
typedef int (__stdcall *IsOpenFn)(void);
static Out32Fn Out32;
static Inp32Fn Inp32;
static LARGE_INTEGER qpf;                   /* QueryPerformanceCounter ticks per second */

static void port_write(int reg, unsigned char v) {
    if (!E.c.dryrun) Out32((short)(E.c.port + reg), (short)v);
}
static unsigned char port_read(int reg) {
    return E.c.dryrun ? 0 : (unsigned char)Inp32((short)(E.c.port + reg));
}
static void spin_us(double us) {
    /* Busy wait (µs) for the step pulse width: Sleep() cannot wait less than ~1 ms. */
    LARGE_INTEGER t0, t;
    double n = us * (double)qpf.QuadPart / 1e6;
    QueryPerformanceCounter(&t0);
    do QueryPerformanceCounter(&t); while ((double)(t.QuadPart - t0.QuadPart) < n);
}
#else
/* Simulator: the «port» is three bytes; inputs come from SIMIN rules on the axis positions. */
static unsigned char sim_reg[3];
typedef struct { int input, axis, greater; long steps; } SimRule;
static SimRule sim_rules[32];
static int sim_nrules;
static int sim_forced[NIN];                 /* -1 = follow the rules; 0/1 = forced with «IN name v» */

static void port_write(int reg, unsigned char v) {
    sim_reg[reg] = v;
    if (E.c.trace >= 2) printf("PORT %llu %d %d\n", E.tick, reg, v);     /* TRACE 2: every register write */
}
static unsigned char port_read(int reg) {
    /* Build the status register from the simulated input levels (inverse of read_inputs). */
    int i, active, bit;
    unsigned char r = 0;
    if (reg != 1) return sim_reg[reg];
    for (i = 0; i < NIN; i++) {
        const Pin *p = &E.c.in[i];
        int k;
        if (!p->pin) continue;
        active = 0;
        if (sim_forced[i] >= 0) active = sim_forced[i];
        else
            for (k = 0; k < sim_nrules; k++)
                if (sim_rules[k].input == i)
                    active |= sim_rules[k].greater ? E.pos[sim_rules[k].axis] > sim_rules[k].steps
                                                   : E.pos[sim_rules[k].axis] < sim_rules[k].steps;
        bit = (active ^ p->activelow) ^ PINS[p->pin].hwinv;
        if (bit) r |= (unsigned char)(1 << PINS[p->pin].bit);
    }
    return r;
}
static void spin_us(double us) { (void)us; }
#endif

static void set_pin(unsigned char *regs, const Pin *p, int active) {
    /* Put logical level «active» on pin p in the register images regs[0..2]. */
    const PinDef *d;
    int bit;
    if (!p->pin) return;
    d = &PINS[p->pin];
    bit = (active ^ p->activelow) ^ d->hwinv;
    if (bit) regs[d->reg] |= (unsigned char)(1 << d->bit);
    else regs[d->reg] &= (unsigned char)~(1 << d->bit);
}

static void write_port(int stepmask) {
    /* Compute data and control bytes (directions, outputs, and the step pins of «stepmask» active)
     * and write only the registers that changed: each port access costs ~1 µs on the ISA/LPC bus. */
    unsigned char regs[3];
    int i;
    regs[0] = 0; regs[1] = 0; regs[2] = 0;
    for (i = 0; i < NAXES; i++) {
        set_pin(regs, &E.c.dir[i], ((E.dirbits >> i) & 1) ^ E.c.dirinv[i]);
        set_pin(regs, &E.c.step[i], (stepmask >> i) & 1);
    }
    for (i = 0; i < NOUT; i++) {
        int on = E.outv[i] != 0;
        if (i == OUT_PWM) {
            /* Software PWM: active during the first duty‰ of each period. */
            double period = E.c.freq / (E.c.pwm_freq > 0 ? E.c.pwm_freq : 100.0);
            on = E.outv[i] > 0 && E.pwm_count < (int)(period * E.outv[i] / 1000.0 + 0.5);
        } else if (i == OUT_CHARGEPUMP) {
            on = E.cp_level;
        } else if (i == OUT_ENABLE) {
            on = E.state != ST_ALARM || !(E.inputs & (1 << IN_ESTOP));
        }
        set_pin(regs, &E.c.out[i], on);
    }
    regs[2] &= 0x0F;                        /* control: keep IRQ (bit 4) and bidirectional (bit 5) off */
    if (!E.written || regs[0] != E.data) { port_write(0, regs[0]); E.data = regs[0]; }
    if (!E.written || regs[2] != E.ctrl) { port_write(2, regs[2]); E.ctrl = regs[2]; }
    E.written = 1;
}

static void read_inputs(void) {
    /* Read the status register once and translate it to logical active inputs. */
    unsigned char st = port_read(1);
    int i, m = 0;
    for (i = 0; i < NIN; i++) {
        const Pin *p = &E.c.in[i];
        int bit;
        if (!p->pin || PINS[p->pin].output) continue;
        bit = (st >> PINS[p->pin].bit) & 1;
        if ((bit ^ PINS[p->pin].hwinv) ^ p->activelow) m |= 1 << i;
    }
    E.inputs = m;
}

/* ==================================================================================================
 * ============================================ MOTION ==============================================
 * ================================================================================================== */
static void start_item(void) {
    /* Take the next queued item (if any) and prepare it. Outputs apply at once and the next item
     * is taken in the same tick, so an «M3» between moves costs no time. */
    while (!E.has_cur && E.qhead != E.qtail) {
        int i, newdir = E.dirbits;
        E.cur = E.q[E.qhead];
        E.qhead = (E.qhead + 1) % QSIZE;
        E.has_cur = 1;
        switch (E.cur.type) {
        case IT_OUT:
            if (E.cur.out >= 0 && E.cur.out < NOUT) E.outv[E.cur.out] = E.cur.value;
            E.last_done = E.cur.id;
            E.has_cur = 0;
            break;
        case IT_DWELL:
            E.dwell_left = E.cur.seconds;
            break;
        case IT_PAUSE:
            E.paused = 1;
            break;
        case IT_MOVE:
            E.s = 0;
            E.probe_hit = 0;
            for (i = 0; i < NAXES; i++) {
                E.done_steps[i] = 0;
                if (E.cur.steps[i] < 0) newdir |= 1 << i;
                else if (E.cur.steps[i] > 0) newdir &= ~(1 << i);
            }
            if (newdir != E.dirbits) {
                E.dirbits = newdir;
                E.dir_wait = 1;             /* direction setup: no step in the same tick as a direction change */
            }
            if (E.cur.probe && (E.inputs & (1 << IN_PROBE))) {
                raise_alarm("la sonda ya estaba activada antes de empezar el sensado");
                return;
            }
            if (E.cur.len <= 0) {           /* nothing to move */
                E.last_done = E.cur.id;
                E.has_cur = 0;
            }
            break;
        }
    }
}

static void finish_move(void) {
    /* The current move ended (fully, or cut by the probe / a home switch). */
    char buf[160];
    if (E.cur.probe) {
        if (!E.probe_hit) {
            if (E.cur.probe == 2) { raise_alarm("sensado sin contacto (G38.2): la sonda no toco nada"); return; }
            snprintf(buf, sizeof buf, "PRB 0 %ld %ld %ld %ld %ld %ld", E.pos[0], E.pos[1], E.pos[2], E.pos[3], E.pos[4], E.pos[5]);
            event(buf);
        }
    } else if (E.cur.home_axis >= 0 && !E.probe_hit) {
        raise_alarm("no encontro el fin de carrera de referencia");
        return;
    }
    E.last_done = E.cur.id;
    E.has_cur = 0;
}

static int motion_tick(double dt) {
    /* Advance the current move by one tick. Returns the mask of axes that must step now. */
    double ov, vc, rem, vmax, a;
    int i, mask = 0;
    Item *m = &E.cur;
    a = m->acc > 0 ? m->acc : 1.0;
    ov = m->rapid ? E.ovr : E.ovf;
    vc = m->vc * ov;
    rem = m->len - E.s;
    vmax = sqrt(m->ve * m->ve + 2.0 * a * (rem > 0 ? rem : 0));     /* braking curve to the exit speed */
    if (E.holding || E.cancelling || E.probe_hit) {
        E.v -= a * dt;
        if (E.v <= 0) {
            E.v = 0;
            if (E.probe_hit || E.cancelling) {
                /* Stopped after the probe / home trigger or a cancel: the rest of the move is dropped. */
                finish_move();
                return 0;
            }
            return 0;                       /* feed hold: stays inside the move */
        }
    } else {
        E.v += a * dt;
        if (E.v > vc) E.v = vc;
        if (E.v > vmax) E.v = vmax;
        /* Floor: near the end of a move that stops, sqrt(2·a·rem) tends to 0 and s would approach L
         * slower and slower; one acceleration step of speed ends it in a finite time. */
        if (E.v < a * dt && vc > 0) E.v = (a * dt < vc) ? a * dt : vc;
    }
    E.s += E.v * dt;
    if (E.s > m->len) E.s = m->len;
    if (E.dir_wait > 0) { E.dir_wait--; return 0; }
    for (i = 0; i < NAXES; i++) {
        long total = labs(m->steps[i]);
        long due = (long)floor((double)total * (E.s / m->len) + 1e-9);
        if (due > total) due = total;
        if (due > E.done_steps[i]) {        /* at most one step per axis per tick */
            mask |= 1 << i;
            E.done_steps[i]++;
            E.pos[i] += m->steps[i] < 0 ? -1 : 1;
        }
    }
    if (E.s >= m->len && !E.probe_hit) {
        int left = 0;
        for (i = 0; i < NAXES; i++) left |= E.done_steps[i] < labs(m->steps[i]);
        if (!left) finish_move();
    }
    return mask;
}

static void tick(void) {
    /* One kernel tick: inputs, safety, motion, outputs. dt is the nominal tick time (see TIMING). */
    double dt = 1.0 / E.c.freq;
    int mask = 0, i;
    char buf[160];
    read_inputs();
    if ((E.inputs & (1 << IN_ESTOP)) && E.state != ST_ALARM) raise_alarm("parada de emergencia");
    if (E.state != ST_ALARM) {
        start_item();
        if (E.has_cur && E.state != ST_ALARM) {
            if (E.cur.type == IT_DWELL) {
                if (!E.holding) E.dwell_left -= dt;
                if (E.dwell_left <= 0 || E.cancelling) { E.last_done = E.cur.id; E.has_cur = 0; }
            } else if (E.cur.type == IT_PAUSE) {
                if (!E.paused || E.cancelling) { E.last_done = E.cur.id; E.has_cur = 0; }
            } else if (E.cur.type == IT_MOVE) {
                /* Limits: any active limit stops everything, except those the move ignores
                 * (backing off a switch) and the home switch of the axis being homed. */
                int lim = (E.inputs >> IN_LIMIT0) & ((1 << NAXES) - 1);
                lim &= ~E.cur.ignore_limits;
                if (E.cur.home_axis >= 0) lim &= ~(1 << E.cur.home_axis);
                if (lim) {
                    for (i = 0; i < NAXES; i++)
                        if (lim & (1 << i)) break;
                    snprintf(buf, sizeof buf, "fin de carrera del eje %c activado", "XYZABC"[i]);
                    raise_alarm(buf);
                } else {
                    if (E.cur.probe && !E.probe_hit && (E.inputs & (1 << IN_PROBE))) {
                        E.probe_hit = 1;    /* latch the position now, brake afterwards */
                        snprintf(buf, sizeof buf, "PRB 1 %ld %ld %ld %ld %ld %ld", E.pos[0], E.pos[1], E.pos[2], E.pos[3], E.pos[4], E.pos[5]);
                        event(buf);
                    }
                    if (E.cur.home_axis >= 0 && !E.probe_hit && (E.inputs & (1 << (IN_HOME0 + E.cur.home_axis)))) {
                        /* Home switch: stop at once (homing seeks are slow) and report. */
                        E.probe_hit = 1;
                        E.v = 0;
                        snprintf(buf, sizeof buf, "HOMED %ld", E.cur.id);
                        event(buf);
                        E.last_done = E.cur.id;
                        E.has_cur = 0;
                    } else {
                        mask = motion_tick(dt);
                    }
                }
            }
        }
        if (E.cancelling && !E.has_cur) {   /* cancel finished braking: drop everything queued */
            E.qhead = E.qtail;
            E.cancelling = 0;
            E.holding = 0;
            E.paused = 0;
        }
        if (E.state != ST_ALARM)
            E.state = (E.holding && E.v == 0) || E.paused ? ST_HOLD : (E.has_cur || E.qhead != E.qtail) ? ST_RUN : ST_IDLE;
    }
    /* Charge pump and PWM counters. */
    if (E.c.out[OUT_CHARGEPUMP].pin && E.c.cp_freq > 0) {
        int half = (int)(E.c.freq / (2.0 * E.c.cp_freq) + 0.5);
        if (half < 1) half = 1;
        if (++E.cp_count >= half) { E.cp_count = 0; E.cp_level ^= 1; }
    }
    if (++E.pwm_count >= (int)(E.c.freq / (E.c.pwm_freq > 0 ? E.c.pwm_freq : 100.0))) E.pwm_count = 0;
    /* Pulses: steps active for PULSE µs, then released. */
    write_port(mask);
    if (mask) {
#if !REAL_PORT
        if (E.c.trace) printf("PULSE %llu %d %d\n", E.tick, mask, E.dirbits);
#endif
        spin_us(E.c.pulse_us);
        write_port(0);
    }
    E.tick++;
}

/* ==================================================================================================
 * =========================================== COMMANDS =============================================
 * ================================================================================================== */
static void push_item(const Item *it) {
    int next = (E.qtail + 1) % QSIZE;
    if (next == E.qhead) { OUT_LOCK(); printf("ERR cola llena\n"); OUT_UNLOCK(); return; }
    E.q[E.qtail] = *it;
    E.qtail = next;
}

static void apply(char *line) {
    /* Apply one command line (see PROTOCOL). Called from the tick thread (real build) or the
     * simulation loop, so it never races with tick(). */
    char w[32], name[32];
    Item it;
    int i;
    if (sscanf(line, "%31s", w) != 1) return;
    memset(&it, 0, sizeof it);
    it.home_axis = -1;
    if (!strcmp(w, "M")) {
        char *p = line + 1, *end;
        double *nums[] = {&it.len, &it.vc, &it.ve, &it.acc};
        it.type = IT_MOVE;
        it.id = strtol(p, &end, 10); p = end;
        for (i = 0; i < 4; i++) { *nums[i] = strtod(p, &end); p = end; }
        it.rapid = (int)strtol(p, &end, 10); p = end;
        for (i = 0; i < NAXES; i++) { it.steps[i] = strtol(p, &end, 10); p = end; }
        while (sscanf(p, "%31s", w) == 1) {
            p = strstr(p, w) + strlen(w);
            if (w[0] == 'P') it.probe = atoi(w + 1);
            else if (w[0] == 'H') it.home_axis = atoi(w + 1);
            else if (w[0] == 'I') it.ignore_limits = atoi(w + 1);
        }
        push_item(&it);
    } else if (!strcmp(w, "O")) {
        int v;
        it.type = IT_OUT;
        if (sscanf(line, "O %ld %31s %d", &it.id, name, &v) == 3) { it.out = out_index(name); it.value = v; push_item(&it); }
    } else if (!strcmp(w, "W")) {
        it.type = IT_DWELL;
        if (sscanf(line, "W %ld %lf", &it.id, &it.seconds) == 2) push_item(&it);
    } else if (!strcmp(w, "P")) {
        it.type = IT_PAUSE;
        if (sscanf(line, "P %ld", &it.id) == 1) push_item(&it);
    } else if (!strcmp(w, "SET")) {
        int v;
        if (sscanf(line, "SET %31s %d", name, &v) == 2 && out_index(name) >= 0) E.outv[out_index(name)] = v;
    } else if (!strcmp(w, "HOLD")) {
        E.holding = 1;
    } else if (!strcmp(w, "RESUME")) {
        E.holding = 0;
        E.paused = 0;
    } else if (!strcmp(w, "CANCEL")) {
        if (E.has_cur || E.qhead != E.qtail) E.cancelling = 1;
        E.paused = 0;
        if (E.has_cur && E.cur.type != IT_MOVE) { E.has_cur = 0; }
        if (!E.has_cur) { E.qhead = E.qtail; E.cancelling = 0; E.holding = 0; }
    } else if (!strcmp(w, "OVF")) {
        double v = atof(line + 3) / 100.0;
        E.ovf = v < 0.1 ? 0.1 : v > 2.0 ? 2.0 : v;
    } else if (!strcmp(w, "OVR")) {
        double v = atof(line + 3) / 100.0;
        E.ovr = v < 0.01 ? 0.01 : v > 1.0 ? 1.0 : v;
    } else if (!strcmp(w, "SETPOS")) {
        if (E.state == ST_RUN) return;
        sscanf(line, "SETPOS %ld %ld %ld %ld %ld %ld", &E.pos[0], &E.pos[1], &E.pos[2], &E.pos[3], &E.pos[4], &E.pos[5]);
    } else if (!strcmp(w, "RESET")) {
        read_inputs();
        if (E.state == ST_ALARM && !(E.inputs & (1 << IN_ESTOP))) E.state = ST_IDLE;
    } else if (!strcmp(w, "IN") && !REAL_PORT) {
#if !REAL_PORT
        int v;
        if (sscanf(line, "IN %31s %d", name, &v) == 2 && in_index(name) >= 0) sim_forced[in_index(name)] = v;
#endif
    } else {
        OUT_LOCK(); printf("ERR comando desconocido: %s\n", w); OUT_UNLOCK();
    }
}

static int config_line(char *line) {
    /* CFG lines (before START). Returns 1 if it was a configuration line. */
    char w[32], name[32];
    int i, pin, low;
    if (sscanf(line, "CFG %31s", w) != 1) return 0;
    if (!strcmp(w, "FREQ")) sscanf(line, "CFG FREQ %lf", &E.c.freq);
    else if (!strcmp(w, "PULSE")) sscanf(line, "CFG PULSE %lf", &E.c.pulse_us);
    else if (!strcmp(w, "PORT")) E.c.port = (unsigned)strtoul(line + 9, NULL, 0);
    else if (!strcmp(w, "REPORT")) sscanf(line, "CFG REPORT %lf", &E.c.report_ms);
    else if (!strcmp(w, "PWMFREQ")) sscanf(line, "CFG PWMFREQ %lf", &E.c.pwm_freq);
    else if (!strcmp(w, "CPFREQ")) sscanf(line, "CFG CPFREQ %lf", &E.c.cp_freq);
    else if (!strcmp(w, "DRYRUN")) sscanf(line, "CFG DRYRUN %d", &E.c.dryrun);
    else if (!strcmp(w, "TRACE")) sscanf(line, "CFG TRACE %d", &E.c.trace);
    else if (!strcmp(w, "TMAX")) sscanf(line, "CFG TMAX %lf", &E.c.tmax);
    else if (!strcmp(w, "INTERACTIVE")) sscanf(line, "CFG INTERACTIVE %lf", &E.c.interactive);
    else if (!strcmp(w, "DLL")) sscanf(line, "CFG DLL %259[^\r\n]", E.c.dll);
    else if (!strcmp(w, "AXIS")) {
        int sp, dp, sl, di;
        if (sscanf(line, "CFG AXIS %d STEP %d DIR %d STEPLOW %d DIRINV %d", &i, &sp, &dp, &sl, &di) == 5 && i >= 0 && i < NAXES
            && sp >= 0 && sp <= 17 && dp >= 0 && dp <= 17 && (!sp || PINS[sp].output) && (!dp || PINS[dp].output)) {
            E.c.step[i].pin = sp; E.c.step[i].activelow = sl;
            E.c.dir[i].pin = dp; E.c.dir[i].activelow = 0; E.c.dirinv[i] = di;
        } else { printf("ERR eje mal configurado: %s\n", line); }
    } else if (!strcmp(w, "OUT")) {
        if (sscanf(line, "CFG OUT %31s %d %d", name, &pin, &low) == 3 && out_index(name) >= 0 && pin >= 0 && pin <= 17
            && (!pin || PINS[pin].output)) {
            E.c.out[out_index(name)].pin = pin; E.c.out[out_index(name)].activelow = low;
        } else { printf("ERR salida mal configurada (pines de salida: 1-9, 14, 16, 17): %s\n", line); }
    } else if (!strcmp(w, "IN")) {
        if (sscanf(line, "CFG IN %31s %d %d", name, &pin, &low) == 3 && in_index(name) >= 0 && pin >= 0 && pin <= 17
            && (!pin || !PINS[pin].output)) {
            E.c.in[in_index(name)].pin = pin; E.c.in[in_index(name)].activelow = low;
        } else { printf("ERR entrada mal configurada (pines de entrada: 10, 11, 12, 13, 15): %s\n", line); }
    } else printf("ERR configuracion desconocida: %s\n", w);
    return 1;
}

static void defaults(void) {
    int i;
    memset(&E, 0, sizeof E);
    E.c.freq = 25000; E.c.pulse_us = 5; E.c.port = 0x378; E.c.report_ms = 20;
    E.c.pwm_freq = 100; E.c.cp_freq = 12500;
    E.ovf = E.ovr = 1.0;
    E.state = ST_IDLE;
    E.last_done = 0;
#if !REAL_PORT
    for (i = 0; i < NIN; i++) sim_forced[i] = -1;
#else
    (void)i;
#endif
}

static void report(void) {
    OUT_LOCK();
    printf("S %c %ld %ld %ld %ld %ld %ld %ld %d %.4f %lu\n", E.state, E.last_done, E.pos[0], E.pos[1], E.pos[2],
           E.pos[3], E.pos[4], E.pos[5], E.inputs, E.v, E.overruns);
    fflush(stdout);
    OUT_UNLOCK();
}

/* ==================================================================================================
 * ====================================== SIMULATOR MAIN LOOP =======================================
 * ================================================================================================== */
#if !REAL_PORT
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
typedef struct { double t; char line[256]; } Timed;

static double wall(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static int sim_rule(char *line) {
    /* «SIMIN name axis >|< steps»: returns 1 if the line was one. */
    char name[32], cmp[4];
    int axis;
    long steps;
    if (strncmp(line, "SIMIN ", 6)) return 0;
    if (sim_nrules < 32 && sscanf(line, "SIMIN %31s %d %3s %ld", name, &axis, cmp, &steps) == 4 && in_index(name) >= 0
        && axis >= 0 && axis < NAXES) {
        sim_rules[sim_nrules].input = in_index(name);
        sim_rules[sim_nrules].axis = axis;
        sim_rules[sim_nrules].greater = cmp[0] == '>';
        sim_rules[sim_nrules].steps = steps;
        sim_nrules++;
    }
    return 1;
}

static int interactive_loop(void) {
    /* Live simulator: every ~2 ms, read the commands that arrived (select, no blocking), run the
     * ticks of the elapsed wall time × speed, print events and the status every REPORT ms. */
    char line[256], buf[8192];
    size_t have = 0;
    double t0 = wall(), next_report = 0;
    int eof = 0;
    setvbuf(stdout, NULL, _IOLBF, 0);
    while (!eof) {
        fd_set fs;
        struct timeval tv = {0, 2000};
        double target;
        FD_ZERO(&fs);
        FD_SET(0, &fs);
        if (select(1, &fs, NULL, NULL, &tv) > 0) {
            ssize_t r = read(0, buf + have, sizeof buf - have - 1);
            if (r <= 0) eof = 1;
            else {
                char *nl;
                have += (size_t)r;
                buf[have] = 0;
                while ((nl = strchr(buf, '\n')) != NULL) {
                    size_t n = (size_t)(nl - buf);
                    if (n > sizeof line - 1) n = sizeof line - 1;
                    memcpy(line, buf, n);
                    line[n] = 0;
                    line[strcspn(line, "\r")] = 0;
                    memmove(buf, nl + 1, have - (size_t)(nl - buf) - 1);
                    have -= (size_t)(nl - buf) + 1;
                    buf[have] = 0;
                    if (!strcmp(line, "QUIT")) { eof = 1; break; }
                    if (!strncmp(line, "TEST", 4)) { printf("LAT 0 0 0 0\n"); continue; }
                    if (line[0] && !sim_rule(line)) apply(line);
                }
            }
        }
        target = (wall() - t0) * E.c.interactive * E.c.freq;
        if (target - (double)E.tick > E.c.freq) target = (double)E.tick + E.c.freq;   /* at most 1 s per round */
        while ((double)E.tick < target) {
            double t = (double)E.tick / E.c.freq;
            tick();
            if (E.c.report_ms > 0 && t >= next_report) { flush_events(); report(); next_report = t + E.c.report_ms / 1000.0; }
        }
        flush_events();
    }
    return 0;
}

int main(void) {
    /* Read everything (CFG, START, «@t» commands), then run virtual ticks until the queue is empty
     * and no timed command is left, or TMAX. Deterministic: the tests compare exact pulse counts. */
    static Timed cmds[4096];
    char line[256];
    int n = 0, k = 0, started = 0, i;
    double t = 0, next_report = 0;
    defaults();
    /* Unbuffered: in INTERACTIVE mode the lines after START are read with read(), and anything
     * stdio had buffered ahead would be lost. */
    setvbuf(stdin, NULL, _IONBF, 0);
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        if (!started && config_line(line)) continue;
        if (!strcmp(line, "START")) {
            started = 1;
            printf("READY %s sim\n", VERSION);
            if (E.c.interactive > 0) { fflush(stdout); return interactive_loop(); }
            continue;
        }
        if (sim_rule(line)) continue;
        if (n < 4096) {
            if (line[0] == '@') { cmds[n].t = atof(line + 1); strcpy(cmds[n].line, strchr(line, ' ') ? strchr(line, ' ') + 1 : ""); }
            else { cmds[n].t = 0; strcpy(cmds[n].line, line); }
            n++;
        }
    }
    if (E.c.tmax <= 0) E.c.tmax = 600;
    for (;;) {
        t = (double)E.tick / E.c.freq;
        while (k < n && cmds[k].t <= t + 1e-12) {
            if (!strncmp(cmds[k].line, "TEST", 4)) printf("LAT 0 0 0 0\n");
            else if (strcmp(cmds[k].line, "QUIT")) apply(cmds[k].line);
            k++;
        }
        tick();
        flush_events();
        if (E.c.report_ms > 0 && t >= next_report) { report(); next_report = t + E.c.report_ms / 1000.0; }
        if ((k >= n && E.state != ST_RUN && !E.has_cur) || t > E.c.tmax) break;
    }
    flush_events();
    report();
    (void)i;
    return 0;
}

/* ==================================================================================================
 * ======================================= WINDOWS MAIN LOOP ========================================
 * ================================================================================================== */
#else
/* Commands go from the reader (main thread) to the tick thread through this single-producer /
 * single-consumer ring: no lock in the tick loop. */
static char cmdq[CMDQ][256];
static volatile LONG cq_head, cq_tail;
static volatile LONG running = 1, test_seconds_x1000 = 0;

static DWORD WINAPI tick_thread(LPVOID arg) {
    /* The real-time loop (see TIMING). */
    LARGE_INTEGER now, next;
    double period;
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    period = (double)qpf.QuadPart / E.c.freq;
    QueryPerformanceCounter(&next);
    while (running) {
        double target;
        while (cq_head != cq_tail) {
            apply(cmdq[cq_head]);
            MemoryBarrier();
            cq_head = (cq_head + 1) % CMDQ;
        }
        if (test_seconds_x1000) {
            /* Latency test: measure how late each tick wakes up, with no motion. */
            unsigned long n = (unsigned long)(E.c.freq * test_seconds_x1000 / 1000.0), i, hist[1001];
            double worst = 0, p99 = 0, p999 = 0, acc = 0;
            memset(hist, 0, sizeof hist);
            QueryPerformanceCounter(&next);
            for (i = 0; i < n; i++) {
                double late;
                target = (double)next.QuadPart + period;
                do QueryPerformanceCounter(&now); while ((double)now.QuadPart < target);
                late = ((double)now.QuadPart - target) * 1e6 / (double)qpf.QuadPart;
                if (late > worst) worst = late;
                hist[late >= 1000 ? 1000 : (int)late]++;
                next.QuadPart = (LONGLONG)target;
            }
            for (i = 0; i <= 1000; i++) {   /* percentiles from the 1 µs histogram */
                acc += hist[i];
                if (!p99 && acc >= 0.99 * n) p99 = i + 1;
                if (!p999 && acc >= 0.999 * n) p999 = i + 1;
            }
            {
                char buf[160];
                snprintf(buf, sizeof buf, "LAT %lu %.1f %.0f %.0f", n, worst, p99, p999);
                event(buf);
            }
            test_seconds_x1000 = 0;
            QueryPerformanceCounter(&next);
        }
        target = (double)next.QuadPart + period;
        do QueryPerformanceCounter(&now); while ((double)now.QuadPart < target);
        if ((double)now.QuadPart - target > period) {
            /* Late by more than one tick: resynchronise instead of catching up with a burst. */
            E.overruns++;
            next = now;
        } else {
            next.QuadPart = (LONGLONG)target;
        }
        tick();
    }
    /* Leaving: outputs off (spindle, coolant), enable released. */
    E.outv[OUT_SPINDLE_CW] = E.outv[OUT_SPINDLE_CCW] = E.outv[OUT_FLOOD] = E.outv[OUT_MIST] = E.outv[OUT_PWM] = 0;
    E.state = ST_ALARM;
    E.inputs |= 1 << IN_ESTOP;
    write_port(0);
    return 0;
}

static DWORD WINAPI report_thread(LPVOID arg) {
    /* Status line every REPORT ms and the queued events. */
    (void)arg;
    while (running) {
        flush_events();
        report();
        Sleep((DWORD)(E.c.report_ms > 1 ? E.c.report_ms : 20));
    }
    return 0;
}

int main(void) {
    char line[256];
    HMODULE dll;
    HANDLE th, rh;
    SYSTEM_INFO si;
    int started = 0;
    defaults();
    InitializeCriticalSection(&out_lock);
    setvbuf(stdout, NULL, _IONBF, 0);
    QueryPerformanceFrequency(&qpf);
    while (!started && fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (config_line(line)) continue;
        if (!strcmp(line, "START")) started = 1;
    }
    if (!started) return 0;
    if (!E.c.dryrun) {
        const char *name = E.c.dll[0] ? E.c.dll : (sizeof(void *) == 8 ? "inpoutx64.dll" : "inpout32.dll");
        IsOpenFn is_open;
        dll = LoadLibraryA(name);
        if (!dll) { printf("ALARM no se encontro %s (driver InpOut del puerto paralelo, va junto a tisma_lpt)\n", name); return 2; }
        /* Copied with memcpy: casting a FARPROC to another function type is what the compiler warns
         * about; the exported signatures are the ones declared above (InpOut's header). */
        {
            FARPROC f;
            f = GetProcAddress(dll, "Out32"); memcpy(&Out32, &f, sizeof f);
            f = GetProcAddress(dll, "Inp32"); memcpy(&Inp32, &f, sizeof f);
            f = GetProcAddress(dll, "IsInpOutDriverOpen"); memcpy(&is_open, &f, sizeof f);
        }
        if (!Out32 || !Inp32) { printf("ALARM %s no es la libreria InpOut esperada\n", name); return 2; }
        if (is_open && !is_open()) {
            printf("ALARM el driver InpOut no esta instalado: ejecutar TISMA CNC una vez como administrador\n");
            return 2;
        }
    }
    /* Priorities: REALTIME needs administrator rights; without them Windows gives HIGH silently. */
    if (!SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    timeBeginPeriod(1);
    write_port(0);
    th = CreateThread(NULL, 0, tick_thread, NULL, 0, NULL);
    GetSystemInfo(&si);
    if (si.dwNumberOfProcessors > 1) SetThreadAffinityMask(th, (DWORD_PTR)1 << (si.dwNumberOfProcessors - 1));
    rh = CreateThread(NULL, 0, report_thread, NULL, 0, NULL);
    printf("READY %s %d %lu\n", VERSION, (int)(sizeof(void *) * 8), (unsigned long)si.dwNumberOfProcessors);
    while (fgets(line, sizeof line, stdin)) {
        LONG next;
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        if (!strcmp(line, "QUIT")) break;
        if (!strncmp(line, "TEST", 4)) { test_seconds_x1000 = (LONG)(atof(line + 4) * 1000); continue; }
        next = (cq_tail + 1) % CMDQ;
        while (next == cq_head) Sleep(1);       /* the tick thread drains it every tick */
        snprintf(cmdq[cq_tail], sizeof cmdq[0], "%s", line);
        MemoryBarrier();
        cq_tail = next;
    }
    /* stdin closed (TISMA ended or crashed): stop the machine and release the port. */
    running = 0;
    WaitForSingleObject(th, 2000);
    WaitForSingleObject(rh, 200);
    timeEndPeriod(1);
    return 0;
}
#endif
