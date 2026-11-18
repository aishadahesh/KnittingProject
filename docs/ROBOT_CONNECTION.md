# Connecting to the UR5e robot

## Network layout

```
University LAN ── Ethernet ──┐
                             PC ── Wi-Fi ── TP-Link router ── LAN cable ── UR5e controller
                                            192.168.0.1                    192.168.0.101 (static)
```

| Device | Address | How it gets it |
|---|---|---|
| TP-Link router | `192.168.0.1` | Router itself; runs DHCP for `192.168.0.0/24` |
| UR5e controller | `192.168.0.101` / `255.255.255.0` | Static, set on the teach pendant |
| PC (Wi-Fi) | `192.168.0.100` (typically) | DHCP from the router |
| PC (Ethernet) | `132.74.121.x` | University DHCP; carries internet |

The PC stays on both networks at once.
Windows routes `192.168.0.x` out the Wi-Fi (on-link route) and everything else out the Ethernet, so the university internet keeps working while connected to the robot.
Never give the Ethernet adapter a `192.168.0.x` address: it steals the Wi-Fi's lease and drops the campus link.

Wi-Fi networks: `TP-Link_CD91` (2.4 GHz) and `TP-Link_CD91_5G` (5 GHz).
Use the 2.4 GHz network; the 5 GHz one has repeatedly hung or lost packets (see *Problems*).

## Robot side (teach pendant)

1. **Settings → System → Network**: Static address, IP `192.168.0.101`, mask `255.255.255.0`, gateway `192.168.0.1`.
2. **Remote control** enabled, and the pendant switched to **Remote** mode (top-right).
   Without Remote, the dashboard server refuses commands such as `play`/`power on`.
3. Robot powered on and brakes released (robot mode `RUNNING`).

## App side

The address is `ur5_robot_ip` in `config/params.json` (default `192.168.0.101`).

| Port | Service |
|---|---|
| 29999 | Dashboard server (mode, safety status, play/stop) |
| 30001 | Primary interface (URScript) |
| 30002 / 30003 | Secondary / real-time interfaces |
| 30004 | RTDE (motion and state streaming in *Direct RTDE* transport) |

Only one RTDE control session is allowed: TeleopApp must not run while the app uses *Direct RTDE*.

## Checking the connection

PowerShell, from the PC:

```powershell
# 1. Wi-Fi is on TP-Link_CD91 with a 192.168.0.x address (169.254.x.x means DHCP failed)
netsh wlan show interfaces | Select-String " SSID|State|Signal"
Get-NetIPAddress -InterfaceAlias Wi-Fi -AddressFamily IPv4

# 2. Router and robot answer with no loss - a single reply is not enough
ping -n 20 192.168.0.1
ping -n 20 192.168.0.101

# 3. Robot ports are open
foreach ($p in 29999,30001,30002,30003,30004) {
  $c = New-Object Net.Sockets.TcpClient
  "$p : $($c.ConnectAsync('192.168.0.101',$p).Wait(2000))"; $c.Close()
}

# 4. Controller state through the dashboard server
$c = New-Object Net.Sockets.TcpClient('192.168.0.101',29999); $s=$c.GetStream()
$r=New-Object IO.StreamReader($s); $w=New-Object IO.StreamWriter($s); $w.AutoFlush=$true
$r.ReadLine()
foreach($cmd in 'is in remote control','robotmode','safetystatus'){ $w.WriteLine($cmd); "$cmd -> $($r.ReadLine())" }
$c.Close()
```

Healthy result: 0% ping loss at a few ms, all five ports `True`, and `is in remote control -> true`, `Robotmode: RUNNING`, `Safetystatus: NORMAL`.

## Problems seen and fixes

### Wi-Fi disconnected
**Symptoms:** `ping 192.168.0.101` fails; `netsh wlan show interfaces` shows `disconnected`, or Windows joined eduroam instead.
**Fix:**
```powershell
netsh wlan connect name="TP-Link_CD91" ssid="TP-Link_CD91" interface="Wi-Fi"
```
Make Windows rejoin 2.4 GHz by itself and never pick 5 GHz:
```powershell
netsh wlan set profileparameter name="TP-Link_CD91" connectionmode=auto
netsh wlan set profileparameter name="TP-Link_CD91_5G" connectionmode=manual
```

### 5 GHz network loses packets or hangs
**Symptoms:** single pings succeed and ports test open, yet RTDE connect fails with `system:10060` (timeout), `system:1236` (connection aborted), `Timeout connecting to UR dashboard server`, or `Could not connect to: 192.168.0.101 at 30004, verify the IP`.
`ping -n 20 192.168.0.1` shows tens of percent loss and up to a second of latency, with a strong signal.
After a full hang the Wi-Fi gets `169.254.x.x` and `ipconfig /renew` cannot reach the DHCP server.
**Cause:** the TP-Link's 5 GHz radio; seen on Nov 11 and Nov 18, 2026 (40–60% loss on Nov 18).
**Fix:** switch to `TP-Link_CD91` (commands above); 2.4 GHz gave 0% loss at 3 ms the same minute.
If 2.4 GHz also hangs, power-cycle the router and reconnect.
If it keeps happening, suspect the router's power supply or firmware, or use a cable (USB-Ethernet adapter to the router).

### "A script sent over the network is running on the arm"
**Symptoms:** connect is refused with this message although TeleopApp is closed and no pendant program is playing.
**Cause:** an earlier RTDE connect uploaded its control script and then lost the link (usually the 5 GHz problem above), leaving the script running.
**Fix:** after confirming nothing else is driving the arm, press the panel's stop-program-and-connect button, or stop the program on the pendant.

### PC still fails after the router is back
**Cause:** Windows kept the `169.254.x.x` fallback address from the outage.
**Fix:** reconnect the Wi-Fi.
A static Wi-Fi address avoids this entirely (admin PowerShell; no gateway, so internet stays on Ethernet):
```powershell
New-NetIPAddress -InterfaceAlias Wi-Fi -IPAddress 192.168.0.50 -PrefixLength 24
Set-NetIPInterface -InterfaceAlias Wi-Fi -Dhcp Disabled
```
Undo with `Set-NetIPInterface -InterfaceAlias Wi-Fi -Dhcp Enabled` and `Remove-NetIPAddress -InterfaceAlias Wi-Fi -IPAddress 192.168.0.50`.

### Router answers but the robot drops out for a few seconds
**Possible causes:** loose LAN cable between router and controller, or an IP clash: `.101` is likely inside the router's DHCP pool (TP-Link default `.100`–`.199`), so a phone or laptop could be handed the robot's address.
**Check:** `Get-NetNeighbor -IPAddress 192.168.0.101`.
The robot's MAC is `00-30-d6-31-1b-3a`; any other MAC there means a clash.
**Fix:** reseat the cable; reserve `.101` in the router's DHCP settings, or move the robot to an address outside the pool (e.g. `192.168.0.20`) on the pendant and update `ur5_robot_ip`.

### Wi-Fi adapter power saving / Intel AX203 quirks
Short, regular stalls on the PC side can come from adapter power saving.
Admin PowerShell:
```powershell
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "802.11n/ac/ax Wireless Mode" -DisplayValue "3. 802.11ac"
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "MIMO Power Save Mode" -DisplayValue "No SMPS"
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "Packet Coalescing" -DisplayValue "Disabled"
Set-NetAdapterAdvancedProperty -Name Wi-Fi -DisplayName "Roaming Aggressiveness" -DisplayValue "1. Lowest"
# Wi-Fi power saving off on battery too
powercfg /setdcvalueindex SCHEME_CURRENT 19cbb8fa-5279-450e-9fac-8a3d5fedd0c1 12bbebe6-58d6-4636-95bb-3217ef867c1a 0
```

### Ports closed but ping works
- Controller still booting, or PolyScope not running: wait and retry.
- Port 30004 refused/busy: another RTDE client (TeleopApp, a stale app process) holds the session; close it.
- Dashboard commands rejected: pendant is in **Local** mode; switch to **Remote**.

### RTDE drops mid-scan
The controller disconnects an RTDE client that falls behind reading the state stream.
The app streams at 125 Hz (`RTDE_FREQUENCY_HZ` in `src/ur5_robot.py`) for this reason; at the 500 Hz default the stream drops over Wi-Fi and a stale pose is returned silently.
Heavy CPU work on the PC during a scan can still starve it, and a lossy Wi-Fi link has the same effect.

### Home move refused
The panel checks the move to Home like any capture move and refuses it when the arm would pass too low over the fabric.
A script driving the arm outside the app must load the saved state (`AppState.load_params(state.load_path, restore_mode=True, ...)`, as `app.py` does), or it checks against a guessed fabric height and the default Home instead of the saved ones.
